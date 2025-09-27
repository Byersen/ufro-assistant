import argparse

def main():
    parser = argparse.ArgumentParser(description="UFRO Assistant CLI")
    parser.add_argument("--provider", choices=["chatgpt", "deepseek"], default="chatgpt")
    parser.add_argument("--query", type=str, help="Pregunta a la normativa UFRO")
    args = parser.parse_args()

    print(f"[DEBUG] Provider={args.provider}, Query={args.query}")

if __name__ == "__main__":
    main()