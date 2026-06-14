import { create } from "zustand";

interface WorkbenchState {
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  selectedTicker: "AAPL",
  setSelectedTicker: (ticker) => set({ selectedTicker: ticker })
}));
