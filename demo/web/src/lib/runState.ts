import type { ResultEvent, RunMeta, ServerEvent, Stage } from "./types";

export type Phase = "idle" | "running" | "evaluating" | "done" | "error";

export interface RunState {
  phase: Phase;
  meta: RunMeta | null;
  stages: Stage[];
  activeStageId: string | null;
  pipeline: { status: string; accepted: boolean; final: string | null; latestDraft: string | null; error: string | null } | null;
  evaluatorText: string;
  evaluatorRunning: boolean;
  result: ResultEvent | null;
  error: string | null;
  startedAt: number | null;
  finishedAt: number | null;
}

export const initialRunState: RunState = {
  phase: "idle",
  meta: null,
  stages: [],
  activeStageId: null,
  pipeline: null,
  evaluatorText: "",
  evaluatorRunning: false,
  result: null,
  error: null,
  startedAt: null,
  finishedAt: null,
};

export type Action = ServerEvent | { type: "reset" } | { type: "begin" } | { type: "client_error"; message: string };

function upsert(stages: Stage[], stage: Stage): Stage[] {
  const index = stages.findIndex((s) => s.stage_id === stage.stage_id);
  if (index < 0) return [...stages, stage];
  const next = stages.slice();
  next[index] = { ...stages[index], ...stage };
  return next;
}

export function runReducer(state: RunState, action: Action): RunState {
  switch (action.type) {
    case "reset":
      return initialRunState;
    case "begin":
      return { ...initialRunState, phase: "running", startedAt: Date.now() };
    case "run_started":
      return { ...state, meta: action.meta };
    case "stage_started":
      return { ...state, stages: upsert(state.stages, action.stage), activeStageId: action.stage.stage_id };
    case "stage_chunk":
      return {
        ...state,
        stages: state.stages.map((s) =>
          s.stage_id === action.stage_id ? { ...s, output: s.output + action.delta } : s,
        ),
      };
    case "stage_completed":
      return { ...state, stages: upsert(state.stages, action.stage) };
    case "pipeline_done":
      return {
        ...state,
        activeStageId: null,
        pipeline: {
          status: action.status,
          accepted: action.accepted,
          final: action.final_explanation,
          latestDraft: action.latest_draft,
          error: action.error,
        },
      };
    case "evaluator_started":
      return { ...state, phase: "evaluating", evaluatorRunning: true, evaluatorText: "" };
    case "evaluator_chunk":
      return { ...state, evaluatorText: state.evaluatorText + action.delta };
    case "result": {
      const { type: _type, ...result } = action;
      return { ...state, evaluatorRunning: false, result };
    }
    case "error":
    case "client_error":
      return { ...state, phase: "error", error: action.message, evaluatorRunning: false, finishedAt: Date.now() };
    case "done":
      return state.phase === "error" ? state : { ...state, phase: "done", finishedAt: Date.now() };
    default:
      return state;
  }
}
