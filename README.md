This is an attempt at fine-tuning an LLM to solve François Chollet's Abstract Reasoning Corpus (ARC) Challenge.

I've fine-tuned the Mistral-7B-Instruct and Llama2-7B-chat-hf pretrained models in these notebooks:
- finetune_Mistral7B_Instruct_on_array_basics.ipynb
- finetune_Llama2_on_array_basics.ipynb

These notebooks can be used to test inference on the fine-tuned models:
- Mistral7B_Instruct_finetuned_for_array_basics.ipynb
- Llama2_finetuned_for_array_basics.ipynb

These notebooks can be used to test inference on the base, pretrained models:
- Mistral7B_Instruct_untuned_for_inference.ipynb
- Llama2_untuned_for_inference.ipynb

So far the fine-tuning is yielding mixed results. The fine-tuned models are correct on the test questions more often than the untuned base models. But the fine-tuned models' answers often continue on with odd text after the answer. I don't yet know why.

Thank you to the people who wrote these blog posts which helped me get started:
- https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8
- https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe
- https://towardsdatascience.com/a-beginners-guide-to-llm-fine-tuning-4bae7d4da672
- https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91
- https://levelup.gitconnected.com/a-step-by-step-guide-to-runing-mistral-7b-ai-on-a-single-gpu-with-google-colab-274a20eb9e40
- https://levelup.gitconnected.com/unleash-mistral-7b-power-how-to-efficiently-fine-tune-a-llm-on-your-own-data-4e4386a6bbdc
- https://blog.gopenai.com/fine-tuning-mistral-7b-instruct-model-in-colab-a-beginners-guide-0f7bebccf11c
- https://medium.com/@mayaakim/complete-guide-to-llm-fine-tuning-for-beginners-c2c38a3252be