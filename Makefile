up:
	docker compose up --build

down:
	docker compose down

ingest:
	docker compose exec api python scripts/ingest.py --path data/docs/sample_docs.jsonl

eval:
	docker compose exec api python scripts/run_eval.py --path data/eval/eval_set.jsonl
