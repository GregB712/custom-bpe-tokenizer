"""Call the tokenizer FastAPI service."""

import httpx


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        print(client.get("/health").json())
        tokenized = client.post(
            "/tokenize",
            json={
                "text": "Tokenization is useful.",
                "tokenizer_type": "bpe",
                "add_special_tokens": True,
            },
        ).json()
        print(tokenized)
        print(
            client.post(
                "/detokenize",
                json={
                    "token_ids": tokenized["token_ids"],
                    "tokenizer_type": "bpe",
                    "skip_special_tokens": True,
                },
            ).json()
        )


if __name__ == "__main__":
    main()
