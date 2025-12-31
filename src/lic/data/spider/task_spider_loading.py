from datasets import load_dataset
from task_spider_eval_old import SpiderEvaluator
from task_spider_process_sql import Schema, get_schema_from_json_obj, get_sql

spider_evaluator = SpiderEvaluator()


def load_spider_seeds(difficulties=["hard", "extra"]):
    dataset = load_dataset("spider")

    spider_schemas = load_dataset("iknow-lab/spider-schema")["train"]
    db_id2raw_schema = {d["db_id"]: d for d in spider_schemas}

    db_id2schema = {
        d["db_id"]: get_schema_from_json_obj(d["tables"]) for d in spider_schemas
    }

    eval_dataset = list(dataset["validation"])
    for i, d in enumerate(eval_dataset):
        d["id"] = f"spider-val-{i}"

    already_sql = set()
    unique_dataset = []
    for d in eval_dataset:
        if d["query"] in already_sql:
            continue

        schema = Schema(db_id2schema[d["db_id"]])
        sql = get_sql(schema, d["query"])

        d["difficulty"] = spider_evaluator.eval_hardness(sql)
        already_sql.add(d["query"])
        unique_dataset.append(d)

    seed_samples = [d for d in unique_dataset if d["difficulty"] in difficulties]

    return seed_samples, db_id2schema, db_id2raw_schema
