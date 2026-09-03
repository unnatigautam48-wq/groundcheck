from evaluate import CORPUS
from pipeline import RAGPipeline


def print_result(result):
    print(f"\nQ: {result.query}")
    print(f"A: {result.answer}")
    print(f"\nGroundedness score: {result.score}  (1.0 = fully backed by sources)")
    for check in result.claim_checks:
        line = f"  [{check.label}] {check.claim}"
        print(line)
        if check.label != "SUPPORTED":
            reason = check.reason or f"closest match only {check.best_match_score} similar"
            print(f"      -> flagged: {reason}")
            print(f"      -> closest source line: \"{check.best_match_source}\"")


def main():
    pipeline = RAGPipeline(CORPUS)
    print("GroundCheck — RAG with a built-in hallucination detector")
    print("Knowledge base: NimbusCloud support docs (pricing, refunds, support, retention, security)")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        result = pipeline.ask(query)
        print_result(result)
        print()


if __name__ == "__main__":
    main()
