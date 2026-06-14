import { create } from "zustand";
import type { Language } from "../i18n";

interface LanguageState {
  language: Language;
  setLanguage: (language: Language) => void;
}

const stored = window.localStorage.getItem("stocktrader-language");
const initialLanguage: Language = stored === "zh" ? "zh" : "en";

export const useLanguageStore = create<LanguageState>((set) => ({
  language: initialLanguage,
  setLanguage: (language) => {
    window.localStorage.setItem("stocktrader-language", language);
    set({ language });
  }
}));
