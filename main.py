"""Command-line chat. Run:  python main.py [--verbose]

--verbose prints each tool call the router chooses (useful for the walkthrough video).
Commands: /reset (new conversation), /quit
"""
import argparse
import sys

import config

from rich.console import Console
from rich.markdown import Markdown


console = Console()

def indexes_ready() -> bool:
    if not config.FLOTATION_CSV.exists():
        print("Note: flotation data not found; flotation analysis is disabled.\n")
    needed = ["brochure_chunks.json", "brochure_vectors.npy", "service_records.json", "service_vectors.npy"]
    return all((config.INDEX_DIR / n).exists() for n in needed)


def main():
    # Windows consoles often use a legacy code page; never crash on a character it cannot show.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="show tool calls")
    args = ap.parse_args()

    if not indexes_ready():
        sys.exit("Indexes not found. Run `python ingest.py` first.")

    from router import Agent   # imported late so the message above appears even if deps are missing
    agent = Agent(verbose=args.verbose)
    print("Metso sales assistant. Type /reset to start over, /quit or /exit to end the session.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit" or text == "/exit":
            break
        if text == "/reset":
            agent = Agent(verbose=args.verbose)
            print("(conversation reset)\n")
            continue
        answer = agent.respond(text)
        console.print("\nAssistant:", style="bold")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()