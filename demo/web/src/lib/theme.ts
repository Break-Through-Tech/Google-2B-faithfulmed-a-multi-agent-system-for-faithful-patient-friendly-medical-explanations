import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const KEY = "fm-theme";
const systemQuery = "(prefers-color-scheme: light)";

function stored(): Theme | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

function system(): Theme {
  return window.matchMedia(systemQuery).matches ? "light" : "dark";
}

/** Follows the system theme until the viewer picks one; index.html applies it before first paint. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? system());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const media = window.matchMedia(systemQuery);
    const onChange = () => {
      if (!stored()) setTheme(system());
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // Private mode or blocked storage: the choice just won't persist.
    }
    setTheme(next);
  };

  return [theme, toggle];
}
