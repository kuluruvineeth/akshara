from dataclasses import dataclass

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


@dataclass(frozen=True)
class EvalSet:
    """One benchmark split, pinned to where it lives.

    `repo` is a Hub dataset (read through its parquet conversion, or from `files` in the repo) or an archive URL
    (read from `files` inside the archive). `columns` keeps only those fields when a table carries other languages.
    """

    name: str
    suite: str
    language: str
    repo: str
    split: str = "test"
    config: str | None = None
    files: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()


EVAL_SETS = (
    # early-signal and base suites
    EvalSet("mmlu", "base", "en", "cais/mmlu", config="all"),
    EvalSet("arc-challenge", "base", "en", "allenai/ai2_arc", config="ARC-Challenge"),
    EvalSet("arc-easy", "base", "en", "allenai/ai2_arc", config="ARC-Easy"),
    EvalSet("hellaswag", "base", "en", "Rowan/hellaswag", split="validation"),
    EvalSet("piqa", "base", "en", "lighteval/piqa", split="validation", config="plain_text"),
    EvalSet("winogrande", "base", "en", "allenai/winogrande", split="validation", config="winogrande_xl"),
    EvalSet("commonsense-qa", "base", "en", "tau/commonsense_qa", split="validation"),
    EvalSet("openbookqa", "base", "en", "allenai/openbookqa", config="main"),
    EvalSet("triviaqa", "base", "en", "mandarjoshi/trivia_qa", split="validation", config="rc.nocontext"),
    EvalSet("gsm8k", "base", "en", "openai/gsm8k", config="main"),
    EvalSet("math", "base", "en", "EleutherAI/hendrycks_math"),
    EvalSet("humaneval-plus", "base", "code", "evalplus/humanevalplus"),
    EvalSet("mbpp-plus", "base", "code", "evalplus/mbppplus"),
    # held out: never tuned on
    EvalSet("mmlu-pro", "held-out", "en", "TIGER-Lab/MMLU-Pro"),
    EvalSet("bbh", "held-out", "en", "lukaemon/bbh"),
    EvalSet("lbpp", "held-out", "code", "CohereLabs/lbpp", config="default"),
    # post-training and reasoning prompts
    EvalSet("ifeval", "sft", "en", "google/IFEval", split="train"),
    EvalSet("ifbench", "sft", "en", "allenai/IFBench_test", split="train"),
    EvalSet("mt-bench", "sft", "en", "HuggingFaceH4/mt_bench_prompts", split="train"),
    EvalSet("alpaca-eval", "sft", "en", "tatsu-lab/alpaca_eval", files=("alpaca_eval.json",)),
    EvalSet("arena-hard", "sft", "en", "lmarena-ai/arena-hard-auto", files=("data/arena-hard-v2.0/question.jsonl",)),
    EvalSet("mixeval-hard-free", "sft", "en", "MixEval/MixEval", split="free_form", config="MixEval_Hard"),
    EvalSet("mixeval-hard-choice", "sft", "en", "MixEval/MixEval", split="multiple_choice", config="MixEval_Hard"),
    EvalSet(
        "bfcl",
        "sft",
        "en",
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        files=tuple(
            f"BFCL_v3_{part}.json"
            for part in (
                "simple",
                "multiple",
                "parallel",
                "parallel_multiple",
                "irrelevance",
                "java",
                "javascript",
                "live_simple",
                "live_multiple",
                "live_parallel",
                "live_parallel_multiple",
                "live_irrelevance",
                "live_relevance",
                "multi_turn_base",
            )
        ),
    ),
    EvalSet("math-500", "reasoning", "en", "HuggingFaceH4/MATH-500"),
    EvalSet("aime25", "reasoning", "en", "math-ai/aime25"),
    # Telugu, Tier S/M
    EvalSet("belebele-te", "telugu", "te", "facebook/belebele", config="tel_Telu"),
    EvalSet("indicqa-te", "telugu", "te", "ai4bharat/IndicQA", files=("data/indicqa.te.json",)),
    EvalSet("indiccopa-te", "telugu", "te", "ai4bharat/IndicCOPA", files=("data/test.te.jsonl",)),
    EvalSet("xstory-cloze-te", "telugu", "te", "juletxara/xstory_cloze", split="eval", config="te"),
    EvalSet("indicxnli-te", "telugu", "te", "Divyanshu/indicxnli", config="te"),
    EvalSet("mmlu-te", "telugu", "te", "lighteval/okapi_mmlu", config="te"),
    EvalSet("arc-te", "telugu", "te", "jon-tow/okapi_arc_challenge", config="te"),
    EvalSet("hellaswag-te", "telugu", "te", "jon-tow/okapi_hellaswag", split="validation", config="te"),
    EvalSet("flores-te", "telugu", "te", FLORES_URL, files=("./flores200_dataset/devtest/tel_Telu.devtest",)),
    EvalSet("flores-en", "telugu", "en", FLORES_URL, files=("./flores200_dataset/devtest/eng_Latn.devtest",)),
    EvalSet("in22-gen", "telugu", "te", "mteb/IN22-Gen", columns=("eng_Latn", "tel_Telu")),
    # Telugu, Tier L+
    EvalSet("mmlu-indic-te", "telugu", "te", "sarvamai/mmlu-indic", config="te"),
    EvalSet("arc-indic-te", "telugu", "te", "sarvamai/arc-challenge-indic", config="te"),
    EvalSet("global-mmlu-te", "telugu", "te", "CohereLabs/Global-MMLU", config="te"),
    EvalSet("indicgenbench-xquad-te", "telugu", "te", "google/IndicGenBench_xquad_in", files=("xquad_te_test.json",)),
    EvalSet("indicgenbench-xorqa-te", "telugu", "te", "google/IndicGenBench_xorqa_in", files=("xorqa_te_test.json",)),
    EvalSet(
        "indicgenbench-crosssum-te",
        "telugu",
        "te",
        "google/IndicGenBench_crosssum_in",
        files=("crosssum_english-te_test.json",),
    ),
    EvalSet(
        "indicgenbench-flores-te",
        "telugu",
        "te",
        "google/IndicGenBench_flores_in",
        files=("flores_en_te_test.json", "flores_te_en_test.json"),
    ),
)

PROMPT_FIELDS = {
    "mmlu": ("question",),
    "arc-challenge": ("question",),
    "arc-easy": ("question",),
    "hellaswag": ("ctx",),
    "piqa": ("goal",),
    "winogrande": ("sentence",),
    "commonsense-qa": ("question",),
    "openbookqa": ("question_stem",),
    "triviaqa": ("question",),
    "gsm8k": ("question",),
    "math": ("problem",),
    "humaneval-plus": ("prompt",),
    "mbpp-plus": ("prompt",),
    "mmlu-pro": ("question",),
    "bbh": ("input",),
    "lbpp": ("instruction",),
    "ifeval": ("prompt",),
    "ifbench": ("prompt",),
    "mt-bench": ("prompt",),
    "alpaca-eval": ("instruction",),
    "arena-hard": ("prompt",),
    "mixeval-hard-free": ("prompt",),
    "mixeval-hard-choice": ("prompt",),
    "bfcl": ("question",),
    "math-500": ("problem",),
    "aime25": ("problem",),
    "belebele-te": ("question",),
    "indiccopa-te": ("premise",),
    "xstory-cloze-te": ("input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4"),
    "indicxnli-te": ("premise", "hypothesis"),
    "mmlu-te": ("question",),
    "arc-te": ("question",),
    "hellaswag-te": ("ctx",),
    "mmlu-indic-te": ("question",),
    "arc-indic-te": ("question",),
    "global-mmlu-te": ("question",),
    "indicgenbench-xquad-te": ("question",),
    "indicgenbench-xorqa-te": ("question",),
    "indicgenbench-crosssum-te": ("text",),
    "indicgenbench-flores-te": ("source",),
}

NOT_FROZEN = {
    "gpqa-diamond": "gated (Idavidrein/gpqa): needs the terms accepted on the Hub",
    "milu-te": "gated (ai4bharat/MILU): needs the terms accepted on the Hub",
    "livecodebench": "4.5 GB, almost all hidden tests; frozen in its own job before the reasoning stage",
    "deepmind-math": "2 GB generator tarball outside the Hub; frozen in its own job before the held-out run",
    "ruler": "synthetic, generated at evaluation time",
    "vlm and speech suites": "images and audio; frozen with their own stages",
}


def by_name(name: str) -> EvalSet:
    return next(eval_set for eval_set in EVAL_SETS if eval_set.name == name)
