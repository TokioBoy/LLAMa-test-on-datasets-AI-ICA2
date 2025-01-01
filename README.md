
# About

The purpose of this project is to test different versions of LLaMa on different benchmarks, such as: ARC-Easy, BoolQ, OpenBookQA, PIQA, TruthfulQA and WinoGender. Primarily the LLaMa 7B and LLaMa 3.2 1B will be tested, but also you will see LLaMa 3B in the tests. If you will need to test another model of LLaMa on some benchmark you can just change the model in the 'model_name' variable and use it.




## Installation

To test any model from LLaMa 3.2 version on the benchmarks by yourself, you will need to receive access to LLaMa's models on a hugginface, here: https://huggingface.co/meta-llama/Llama-3.2-1B

Then, you should install all necessary dependencies, which are quite common for LLM's testing, such as PyTorch, Transformers, Datasets, tqdm, numpy and others. It is recommended to use IDE and check what is required for each test for a particular benchmark and if you dont have it install it through IDE. Another way is to install all dependencies through 'pip' or 'conda' commands, in that case all necessary commands you can find on official pages.

When you will have, if needed, access to a model, which you wanna test and will install all dependencies you can run a test. If benchmarks or model is not installed yet on your device, instalation will start and when it will be finished, testing process will begin. 

NOTE: if nothing happens after you run code, and you dont see any errors most likely it means that benchmark or a model is loading, you should just wait. If you will encounter some error, what you cannot solve, please, contact support.

You can find the results of your test in the console, when the test will finish or in .csv file, if it is intended.



    
## Acknowledgements

 - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)

 - [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://arxiv.org/pdf/1905.10044)

- [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/pdf/1809.02789)

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/pdf/2109.07958)

- [Gender Bias in Coreference Resolution](https://arxiv.org/pdf/1804.09301)

- [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/pdf/1803.05457)

- [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/pdf/1911.11641)

## Authors

- [@TokioBoy](https://github.com/TokioBoy)


## Support

For support, email vladyslav.zolotarevskyi@praguecityuniversity.cz

