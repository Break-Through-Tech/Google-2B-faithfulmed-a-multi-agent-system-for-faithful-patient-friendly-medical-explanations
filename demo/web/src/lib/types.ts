export type StageStatus = "running" | "completed" | "failed" | "skipped";

export interface Stage {
  stage_id: string;
  agent: string;
  iteration: number | null;
  status: StageStatus;
  started_at_utc: string | null;
  completed_at_utc: string | null;
  output: string;
  parsed_output: any;
  error: string | null;
}

export interface Span {
  span_id: string;
  text: string;
}

export type Classification = "supported" | "unsupported" | "contradicted" | "uncertain";

export interface Claim {
  claim_id: string;
  claim_span_id: string | null;
  claim_text: string;
  classification: Classification;
  clinically_meaningful: boolean;
  source_span_ids: string[];
  reason: string;
}

export interface ResultEvent {
  readability: {
    flesch_kincaid_grade: number;
    smog_index: number;
    jargon_density: number;
    word_count: number;
    meets_target: boolean;
    is_refusal: boolean;
  } | null;
  readability_error: string | null;
  source_grade: number | null;
  factual: {
    claims_assessed: number;
    claims_returned: number;
    supported_claims: number;
    uncertain_claims: number;
    factual_fidelity_estimate: number | null;
    clinically_meaningful_hallucination: boolean | null;
  } | null;
  evaluator_status: string | null;
  evaluator_status_reason: string | null;
  evaluator_error: string | null;
  source_spans: Span[];
  explanation_spans: Span[];
  claims: Claim[];
}

export interface RunMeta {
  mode: "live" | "replay";
  run_id?: string;
  dataset: string;
  split: string;
  example_id: string;
  model: string;
  recorded_at_utc?: string;
  source_text: string;
}

export interface SavedRun {
  run_id: string;
  dataset: string;
  split: string;
  example_id: string;
  created_at_utc: string;
  model: string;
  revisions: number;
  fk_grade: number | null;
  fidelity: number | null;
  hallucination: boolean | null;
  preview: string;
}

export interface LiveExample {
  key: string;
  dataset: string;
  split: string;
  example_id: string;
  chars: number;
  source_text: string;
}

export interface Config {
  live_available: boolean;
  model: string;
  max_revision_cycles: number;
  max_source_chars: number;
}

export type ServerEvent =
  | { type: "run_started"; meta: RunMeta }
  | { type: "stage_started"; stage: Stage }
  | { type: "stage_chunk"; stage_id: string; delta: string }
  | { type: "stage_completed"; stage: Stage }
  | {
      type: "pipeline_done";
      status: string;
      accepted: boolean;
      final_explanation: string | null;
      latest_draft: string | null;
      error: string | null;
    }
  | { type: "evaluator_started" }
  | { type: "evaluator_chunk"; delta: string }
  | ({ type: "result" } & ResultEvent)
  | { type: "error"; message: string }
  | { type: "done" };
