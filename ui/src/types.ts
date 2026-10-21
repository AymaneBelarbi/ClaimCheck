// Field names mirror src/claimcheck/core/review.py, src/claimcheck/api/schemas.py and the /ask response exactly.

export type Verdict = "found" | "missing" | "conflicting" | "low_confidence";
export type ClaimType = "collision" | "glass" | "theft";
export type DocType = "claim_form" | "id_card" | "invoice" | "incident_report" | "other";

export const CLAIM_STATUSES = [
  "received",
  "processing",
  "needs_review",
  "approved_for_review",
  "rejected_incomplete",
  "failed",
] as const;
export type ClaimStatus = (typeof CLAIM_STATUSES)[number];

export interface ClaimSummary {
  claim_id: string;
  packet_id: string;
  status: ClaimStatus;
  claim_type: ClaimType;
  incident_date: string;
  created_at: string;
}

export interface Evidence { document_id: string; page: number | null; span: [number, number] | null; text: string | null; }

export interface ReviewClaim { id: string; packet_id: string; status: ClaimStatus; claim_type: ClaimType; incident_date: string; }

export interface ReviewPolicyVersion { version_no: number; effective_from: string; effective_to: string | null; }

export interface ReviewDocument {
  id: string;
  filename: string;
  doc_type: DocType | null;
  classification_confidence: number | null;
  classification_method: string | null;
  ocr_mean_confidence: number | null;
  low_confidence: boolean;
  error: string | null;
}

export interface ReviewField {
  document_id: string;
  doc_type: DocType | null;
  field_name: string;
  value: string | null;
  confidence: number;
  verdict: Verdict;
  evidence: Evidence | null;
}

export interface ReviewRule {
  rule_id: string;
  description: string;
  verdict: Verdict;
  message: string;
  evidence: Evidence[];
}

export interface ReviewSummary {
  verdict_counts: Record<Verdict, number>;
  field_verdict_counts: Record<Verdict, number>;
  blocking: string[];
  ready_for_decision: boolean;
}

export interface ReviewFile {
  claim: ReviewClaim;
  policy_version: ReviewPolicyVersion | null;
  documents: ReviewDocument[];
  fields: ReviewField[];
  rules: ReviewRule[];
  summary: ReviewSummary;
}

export interface AskAnswer { answer: string; citations: string[]; uncited: boolean; tools_called: string[]; }

export type Decision = "approved_for_review" | "rejected_incomplete";
