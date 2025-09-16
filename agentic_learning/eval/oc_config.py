from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_a0fc46 import sanitized_mbpp_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset 
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import QUERY_TEMPLATE

from opencompass.datasets.aime2024 import LocalAime24Dataset
from opencompass.datasets.aime2025 import LocalAime25Dataset
from opencompass.datasets.math import LocalMATH500Dataset
from opencompass.datasets.livemathbench.livemathbench import LocalLiveMathBenchDataset
from opencompass.datasets.mmlu_pro import LocalMMLUProDataset
from opencompass.datasets.gpqa import LocalGPQADiamondDataset, GPQAEvaluator, GPQA_Eval_postprocess, LocalReasoningGymDataset
from opencompass.datasets.mbpp import LocalSanitizedMBPPDataset
from opencompass.datasets.livecodebench.livecodebench import LocalLCBCodeGenerationV6Dataset
from opencompass.datasets.livecodebench.evaluator import LocalLCBCodeGenerationV6Evaluator

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.tasks import OpenICLInferTask
from opencompass.models import TurboMindModelwithChatTemplate, VLLMwithChatTemplate
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner


aime24 = dict(
    type=LocalAime24Dataset,
    n=16,
    abbr="aime24",
    reader_cfg=dict(
        input_columns=["question"], 
        output_column="answer"
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role="HUMAN", 
                        prompt="{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                    ),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHVerifyEvaluator
        )
    )
)

aime25 = dict(
    type=LocalAime25Dataset,
    n=16,
    abbr="aime25",
    reader_cfg=dict(
        input_columns=["question"], 
        output_column="answer"
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt="{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHVerifyEvaluator
        )
    )
)

math500 = dict(
    type=LocalMATH500Dataset,
    n=4,
    abbr="math500",
    reader_cfg=dict(
        input_columns=["question"], 
        output_column="answer"
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt="{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHVerifyEvaluator
        )
    )
)

livemathbench = dict(
    type=LocalLiveMathBenchDataset,
    n=4,
    abbr="livemathbench",
    reader_cfg=dict(
        input_columns=["question"], 
        output_column="answer"
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt="{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHVerifyEvaluator
        )
    )
)

mmlu_pro = dict(
    type=LocalMMLUProDataset,
    n=4,
    abbr="mmlu_pro",
    reader_cfg=dict(
        input_columns=["question", "cot_content", "options_str"],
        output_column="answer",
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN',
                         prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=GPQAEvaluator
        ),
        pred_postprocessor=dict(
            type=GPQA_Eval_postprocess
        )
    )
)

gpqa_dataset = gpqa_datasets[0]
gpqa_dataset.pop("path")
gpqa_dataset.pop("name")
gpqa_dataset.update(
    dict(
        n=4,
        type=LocalGPQADiamondDataset
    )
)
gpqa_dataset["eval_cfg"]["pred_postprocessor"].update(
    type=GPQA_Eval_postprocess
)

mbpp_dataset = sanitized_mbpp_datasets[0]
mbpp_dataset.pop("path")
mbpp_dataset.update(
    type=LocalSanitizedMBPPDataset,
    n=4,
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\n',),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=32768),
    )
)

lcb_dataset = LCBCodeGeneration_dataset
lcb_dataset.pop("path")
lcb_dataset.update(
    dict(
        type=LocalLCBCodeGenerationV6Dataset,
        n=4, 
        abbr="livecodebench_v6"
    )
)
lcb_dataset["eval_cfg"].update(
    dict(
        evaluator=dict(
            type=LocalLCBCodeGenerationV6Evaluator,
            num_process_evaluate=4,
            timeout=6,
            extractor_version="v2",
        )
    )
)

reasoning_gym = dict(
    type=LocalReasoningGymDataset,
    n=4,
    abbr="reasoning_gym",
    reader_cfg=dict(
        input_columns=["question"], 
        output_column="answer"
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role="HUMAN", 
                        prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
                    ),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=GPQAEvaluator
        ),
        pred_postprocessor=dict(
            type=GPQA_Eval_postprocess
        )
    )
)

datasets = [aime24, aime25, math500, livemathbench, mmlu_pro, gpqa_dataset, mbpp_dataset, lcb_dataset, reasoning_gym]

models = [
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="Qwen/Qwen2.5-7B-Instruct",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="qwen2_5-7b",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # ),
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="qwen3-30b",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # ),
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/short-cot-distillation/qwen3-30b_2_qwen3-4b/global_step_1509",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="al-sft",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # ),
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-NativeAll-Qwen2.5-7B-Instruct-20250907_135543/global_step_239/actor/checkpoint",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="al-native-all",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # ),
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-All-Qwen2.5-7B-Instruct-20250907_133154/global_step_239/actor/checkpoint",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="al-mask-all",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # ),
    # dict(
    #     type=VLLMwithChatTemplate,
    #     path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-All-Qwen2.5-7B-Instruct-20250912_194233/global_step_100/actor/checkpoint",
    #     meta_template=dict(
    #         round=[
    #             dict(role='HUMAN', api_role='HUMAN'),
    #             dict(role='BOT', api_role='BOT', generate=True),
    #         ]
    #     ),
    #     abbr="al",
    #     model_kwargs=dict(tensor_parallel_size=1),
    #     generation_kwargs=dict(
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1
    #     ),
    #     max_out_len=16384,
    #     batch_size=256,
    #     run_cfg=dict(num_gpus=1)
    # )
    dict(
        type=VLLMwithChatTemplate,
        path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-All-Qwen2.5-7B-Instruct-20250915_013829/global_step_200/actor/checkpoint",
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        abbr="al-200",
        model_kwargs=dict(tensor_parallel_size=1),
        generation_kwargs=dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1
        ),
        max_out_len=32768,
        batch_size=256,
        run_cfg=dict(num_gpus=1)
    ),
    dict(
        type=VLLMwithChatTemplate,
        path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-All-Qwen2.5-7B-Instruct-20250915_013829/global_step_220/actor/checkpoint",
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        abbr="al-220",
        model_kwargs=dict(tensor_parallel_size=1),
        generation_kwargs=dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1
        ),
        max_out_len=32768,
        batch_size=256,
        run_cfg=dict(num_gpus=1)
    ),
    dict(
        type=VLLMwithChatTemplate,
        path="/mnt/shared-storage-user/liujunnan/project/AgenticLearning/checkpoints/RL/RL-All-Qwen2.5-7B-Instruct-20250915_013829/global_step_239/actor/checkpoint",
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        abbr="al-239",
        model_kwargs=dict(tensor_parallel_size=1),
        generation_kwargs=dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1
        ),
        max_out_len=32768,
        batch_size=256,
        run_cfg=dict(num_gpus=1)
    ),
    dict(
        type=VLLMwithChatTemplate,
        path="meta-llama/Llama-3.2-3B-Instruct",
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        abbr="llama-3_2-3b-it",
        model_kwargs=dict(tensor_parallel_size=1),
        generation_kwargs=dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1
        ),
        max_out_len=32768,
        batch_size=256,
        run_cfg=dict(num_gpus=1)
    ),
]
# /mnt/shared-storage-user/liujunnan/project/RePro/checkpoints/repro/DeepSeek-R1-Distill-Qwen-1.5B_deepscale-r-preview_16k_20250901_120908/global_step_440/actor/huggingface

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    
        num_split=None,   
        min_task_size=64, 
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    )
)