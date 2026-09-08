"""Single local coordinator, independent of the HTTP process and browser sessions."""
import argparse
import logging
import multiprocessing
from pathlib import Path
import os
import signal
from threading import Event, Thread
import time
from uuid import uuid4

from filelock import FileLock, Timeout

from ..artifacts import ArtifactPublisher
from ..errors import AppError
from ..models import ErrorSummary, JobRecord
from ..settings import AppSettings, confined_path
from ..storage.database import Database
from ..storage.jobs import JobRepository, TERMINAL_STATUSES
from .compute import compute_entry

LOG = logging.getLogger(__name__)


def map_event(event: dict, job: JobRecord) -> dict:
    """Translate core observations, without changing research decisions or counting twice."""
    name = event.get('event')
    if name == 'phase':
        return {'phase': event['phase'], **({'seed': event['seed']} if 'seed' in event else {})}
    if name == 'seed_start':
        return {'seed': event['seed'], 'seed_index': event['seed_index']}
    if name == 'training':
        return {'phase': 'training', 'seed': event['seed'], 'actual_steps': event['timesteps']}
    if name == 'seed_complete':
        updates = {'seed': event['seed'], 'seed_index': event['seed_index'],
                   'completed_seeds': max(job.completed_seeds, event['seed_index'] + 1)}
        if job.kind == 'train':
            updates['actual_steps'] = event['timesteps']
        return updates
    return {}


class Worker:
    def __init__(self, settings: AppSettings, *, stop_file: Path | None = None):
        self.settings, self.stop_file = settings, stop_file
        self.db = Database(settings.database_path, settings.busy_timeout_ms)
        self.jobs = JobRepository(self.db, settings.worker_unavailable_seconds)
        self.publisher = ArtifactPublisher(settings, self.db)
        self.owner = str(uuid4())
        self.stopping = Event()
        self._last_heartbeat = float('-inf')
        self.context = multiprocessing.get_context('spawn')

    def stop_requested(self) -> bool:
        return self.stopping.is_set() or bool(self.stop_file and self.stop_file.exists())

    def heartbeat(self, *, active_job_id: str | None = None, recovery_waiting: bool = False,
                  force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_heartbeat >= self.settings.heartbeat_interval_seconds:
            self.jobs.heartbeat(self.owner, active_job_id=active_job_id,
                                recovery_waiting=recovery_waiting, pid=os.getpid())
            self._last_heartbeat = now

    def recover(self) -> bool:
        """Never interpret a dead coordinator as proof that its computation has stopped."""
        execution = FileLock(confined_path(self.settings.app_dir, 'locks/execution.lock'), timeout=0)
        while not self.stop_requested():
            try:
                execution.acquire()
                break
            except Timeout:
                self.heartbeat(recovery_waiting=True)
                self.stopping.wait(.2)
        else:
            return False
        try:
            with self.db.connection() as connection:
                active = connection.execute("SELECT job_id FROM jobs WHERE status IN ('running','cancelling')").fetchall()
            for row in active:
                job = self.jobs.get(row['job_id'])
                try:
                    if self.publisher.recover_published(job, check=lambda: self.heartbeat(recovery_waiting=True)):
                        continue
                except AppError as exc:
                    if exc.status_code >= 500:
                        raise
                    LOG.exception('Quarantined publication for job %s', job.job_id)
                # Recovery owns the OS execution lock, and checks the old persisted token/revision.
                # It does not forge new ownership or touch an unknown process identifier.
                job = self.jobs.get(job.job_id)
                if job.status in TERMINAL_STATUSES:
                    continue
                self.jobs.transition(job.job_id, 'interrupted', owner_token=job.owner_token,
                    revision=job.revision, error=ErrorSummary(code='WORKER_INTERRUPTED',
                    message='执行进程已结束，任务未完整发布；保留部分产物，不自动重试。'))
            self.heartbeat(force=True)
            return True
        finally:
            execution.release()

    @staticmethod
    def send_cancel(connection) -> None:
        try:
            connection.send('cancel')
        except (OSError, BrokenPipeError):
            pass

    def finish_stopped(self, job_id: str, *, timed_out: bool, error: ErrorSummary | None) -> None:
        for _ in range(3):
            job = self.jobs.get(job_id)
            if job.status in TERMINAL_STATUSES:
                return
            if timed_out:
                status, failure = 'failed', ErrorSummary(code='TIME_LIMIT_EXCEEDED', message='任务超过总时限，执行已停止。')
            elif job.status == 'cancelling':
                status, failure = 'cancelled', None
            else:
                status, failure = 'failed', error or ErrorSummary(code='COMPUTE_INTERRUPTED', message='计算进程提前结束，结果未发布。')
            try:
                self.jobs.transition(job_id, status, owner_token=self.owner, revision=job.revision, error=failure)
                return
            except AppError as exc:
                if exc.code != 'STATE_CONFLICT':
                    raise
        raise AppError('STATE_CONFLICT', '任务状态持续变化，等待下次恢复确认。', 409)

    def publish(self, job: JobRecord, start: float) -> None:
        finished = Event()
        last_state_check = float('-inf')

        def keep_alive():
            while not finished.wait(self.settings.heartbeat_interval_seconds):
                try:
                    self.jobs.heartbeat(self.owner, active_job_id=job.job_id, pid=os.getpid())
                except AppError:
                    # Main publication still performs authoritative DB checks. A final-state
                    # race here is harmless and a DB failure cannot authorize publication.
                    return

        def check_deadline():
            nonlocal last_state_check
            now = time.monotonic()
            if now - start >= self.settings.task_time_limit_seconds:
                raise AppError('TIME_LIMIT_EXCEEDED', '任务超过总时限，结果未发布。', 409)
            if self.stop_requested():
                raise AppError('WORKER_STOPPING', '执行服务正在退出，结果未发布。', 409)
            if now - last_state_check >= .25:
                latest = self.jobs.get(job.job_id)
                if latest.status != 'running' or latest.cancel_requested or latest.owner_token != self.owner:
                    raise AppError('STATE_CONFLICT', '任务已停止或取消，结果未发布。', 409)
                last_state_check = now

        thread = Thread(target=keep_alive, name='publishing-heartbeat', daemon=True)
        thread.start()
        try:
            self.publisher.publish(job, check_deadline=check_deadline)
        finally:
            finished.set()
            thread.join(self.settings.busy_timeout_ms / 1000 + 1)

    def run_job(self, initial: JobRecord) -> None:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(target=compute_entry,
            args=(self.settings, initial.job_id, self.owner, child), name=f'stockrl-{initial.job_id}')
        started = False
        done, timed_out, stopping_at = False, False, None
        pipe_closed = False
        terminated, killed, reported_unconfirmed = False, False, False
        error = None
        start = time.monotonic()
        try:
            process.start()
            started = True
            child.close()
            while True:
                now = time.monotonic()
                job = self.jobs.get(initial.job_id)
                timed_out = timed_out or now - start >= self.settings.task_time_limit_seconds
                if (self.stop_requested() or timed_out) and job.status == 'running':
                    job = self.jobs.request_cancel(job.job_id)
                if stopping_at is None and (timed_out or job.status == 'cancelling'):
                    stopping_at = now
                    self.send_cancel(parent)
                # Drain bounded batches so a noisy child cannot starve cancellation/heartbeats.
                for _ in range(100):
                    if pipe_closed or not parent.poll():
                        break
                    try:
                        message = parent.recv()
                    except (EOFError, OSError):
                        pipe_closed = True
                        break
                    if message.get('type') == 'done':
                        done = True
                    elif message.get('type') == 'error':
                        error = ErrorSummary.model_validate(message['error'])
                    elif message.get('type') == 'event' and job.status == 'running' and not timed_out:
                        updates = map_event(message['event'], job)
                        if updates:
                            try:
                                job = self.jobs.update_progress(job.job_id, owner_token=self.owner,
                                                               revision=job.revision, **updates)
                            except AppError as exc:
                                if exc.code != 'STATE_CONFLICT':
                                    raise
                                job = self.jobs.get(job.job_id)
                alive = process.is_alive()
                if not alive:
                    process.join()
                    # The last pipe message can become readable between the poll and process exit.
                    if not pipe_closed and parent.poll():
                        continue
                    break
                waiting = False
                if stopping_at is not None:
                    elapsed = now - stopping_at
                    grace = self.settings.cancellation_grace_seconds
                    confirmation = self.settings.termination_wait_seconds
                    if elapsed >= grace and not terminated:
                        process.terminate()
                        terminated = True
                    if elapsed >= grace + confirmation / 2 and not killed:
                        process.kill()
                        killed = True
                    waiting = elapsed >= grace + confirmation
                    if waiting and not reported_unconfirmed:
                        try:
                            self.jobs.append_event(job.job_id, 'error', ErrorSummary(
                                code='TERMINATION_UNCONFIRMED', message='尚未确认计算停止，保持取消中并暂停新任务。'),
                                owner_token=self.owner, revision=job.revision)
                            reported_unconfirmed = True
                        except AppError as exc:
                            if exc.code != 'STATE_CONFLICT':
                                raise
                    # A still-live child keeps its lock and this coordinator; never claim another job.
                self.heartbeat(active_job_id=job.job_id, recovery_waiting=waiting)
                time.sleep(.2)
            job = self.jobs.get(initial.job_id)
            if done and process.exitcode == 0 and job.status == 'running' and not timed_out:
                try:
                    self.publish(job, start)
                    return
                except AppError as exc:
                    if exc.status_code >= 500:
                        raise
                    # Cancellation may have committed between verify, intent and final transaction.
                    error = ErrorSummary(code=exc.code, message=exc.message)
                    timed_out = timed_out or exc.code == 'TIME_LIMIT_EXCEEDED'
                    if exc.code == 'WORKER_STOPPING':
                        self.jobs.request_cancel(job.job_id)
                    LOG.exception('Publication rejected for job %s', job.job_id)
            self.finish_stopped(initial.job_id, timed_out=timed_out, error=error)
        finally:
            # Even a database error must not release coordinator ownership while its child lives.
            if started and process.is_alive():
                self.send_cancel(parent)
                process.join(self.settings.cancellation_grace_seconds)
                if process.is_alive():
                    process.terminate()
                    process.join(self.settings.termination_wait_seconds / 2)
                if process.is_alive():
                    process.kill()
                    process.join(self.settings.termination_wait_seconds / 2)
                while process.is_alive():
                    LOG.error('Waiting for owned compute process to exit; no new jobs will start')
                    process.join(1)
            parent.close()
            child.close()
            if started:
                process.close()

    def run(self) -> int:
        lock_dir = confined_path(self.settings.app_dir, 'locks')
        lock_dir.mkdir(parents=True, exist_ok=True)
        # Opening an initialized database is explicit; workers never migrate it.
        with self.db.connection():
            pass
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(lock_dir / 'coordinator.lock', timeout=0):
                if not self.recover():
                    return 0
                while not self.stop_requested():
                    self.heartbeat()
                    job = self.jobs.claim_next(self.owner)
                    if job is None:
                        self.stopping.wait(.2)
                        continue
                    self.heartbeat(active_job_id=job.job_id, force=True)
                    self.run_job(job)
                    self.heartbeat(force=True)
                return 0
        except Timeout:
            LOG.error('另一个 worker 已经运行；本实例没有领取任务。')
            return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='StockRL 本地单任务执行服务')
    parser.add_argument('--stop-file', type=Path, help='本地启动器写入此文件后停止领取并取消当前任务')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    worker = Worker(AppSettings.from_env(), stop_file=args.stop_file)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: worker.stopping.set())
    try:
        return worker.run()
    except (AppError, OSError):
        LOG.exception('Worker stopped; persisted tasks will be inspected at the next startup')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
