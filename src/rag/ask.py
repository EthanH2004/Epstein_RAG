def main():
    print("RAG CLI (stub). Type 'exit' to quit.")
    while True:
        q = input("> ")
        if q.strip().lower() in {"exit", "quit"}:
            break

        # TODO: embed question -> retrieve -> LLM answer
        print(f"[stub] You asked: {q}")

if __name__ == "__main__":
    main()