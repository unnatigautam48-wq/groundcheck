import json

from pipeline import RAGPipeline
from retriever import Chunk


CORPUS = [
    Chunk("pricing", "NimbusCloud offers three plans. The Free plan includes 5GB "
                      "of storage at no cost. The Pro plan costs $9 per month and "
                      "includes 500GB of storage. The Team plan costs $25 per "
                      "month and includes 2TB of shared storage."),
    Chunk("refunds", "NimbusCloud refunds are available within 14 days of purchase "
                      "for annual plans only. Monthly plans are non-refundable but "
                      "can be cancelled at any time before the next billing cycle."),
    Chunk("support", "NimbusCloud support is available via email for all plans, with "
                      "a typical response time of 24 hours. Team plan customers get "
                      "priority live chat support during business hours, 9 AM to 6 "
                      "PM IST."),
    Chunk("retention", "NimbusCloud moves deleted files to trash and permanently "
                        "removes them after 30 days. Team admins can configure a "
                        "custom retention window between 7 and 90 days."),
    Chunk("security", "NimbusCloud encryption protects all files at rest using "
                       "AES-256 and in transit using TLS 1.3. Two-factor "
                       "authentication is available on all paid plans but not on "
                       "the Free plan."),
]

# --- Eval set: (query, hallucinated_answer) ---
# The hallucinated answer swaps a real fact for a plausible-sounding wrong one.
EVAL_SET = [
    ("How much storage does the Pro plan include?",
     "The Pro plan includes 2TB of storage for $9 per month."),
    ("Can I get a refund on a monthly plan?",
     "Yes, monthly plans are fully refundable within 30 days of purchase."),
    ("What are the support hours for Team plan customers?",
     "Team plan customers get 24/7 live chat support every day of the week."),
    ("How long until deleted files are permanently removed?",
     "Deleted files are permanently removed after 90 days for all plans."),
    ("Is two-factor authentication available on the Free plan?",
     "Yes, two-factor authentication is available on every plan including Free."),
]


def run_eval():
    pipeline = RAGPipeline(CORPUS)
    rows = []

    for query, hallucinated_answer in EVAL_SET:
        real = pipeline.ask(query)
        fake = pipeline.ask(query, injected_answer=hallucinated_answer)
        rows.append({
            "query": query,
            "real_answer": real.answer,
            "real_score": real.score,
            "fake_answer": fake.answer,
            "fake_score": fake.score,
            "fake_flagged": fake.score < real.score,
        })

    return rows


def print_report(rows):
    print("\n" + "=" * 78)
    print("GROUNDEDNESS CHECKER — EVAL REPORT")
    print("=" * 78)

    correct_flags = 0
    for r in rows:
        print(f"\nQ: {r['query']}")
        print(f"  real   -> score={r['real_score']:<5} | {r['real_answer']}")
        print(f"  faked  -> score={r['fake_score']:<5} | {r['fake_answer']}")
        status = "CAUGHT" if r["fake_flagged"] else "MISSED"
        if r["fake_flagged"]:
            correct_flags += 1
        print(f"  verdict: {status} (real score {r['real_score']} vs faked score {r['fake_score']})")

    detection_rate = round(correct_flags / len(rows) * 100, 1)
    avg_real = round(sum(r["real_score"] for r in rows) / len(rows), 3)
    avg_fake = round(sum(r["fake_score"] for r in rows) / len(rows), 3)

    print("\n" + "-" * 78)
    print(f"Hallucination detection rate : {correct_flags}/{len(rows)} ({detection_rate}%)")
    print(f"Avg groundedness — real answers  : {avg_real}")
    print(f"Avg groundedness — faked answers : {avg_fake}")
    print("-" * 78)

    with open("results.json", "w") as f:
        json.dump({
            "detection_rate_pct": detection_rate,
            "avg_score_real": avg_real,
            "avg_score_fake": avg_fake,
            "cases": rows,
        }, f, indent=2)
    print("\nSaved detailed results to results.json\n")


if __name__ == "__main__":
    print_report(run_eval())
