import model_interactor

test = model_interactor.DatasetInteractor()

data = test.select_prompts_sample()
for d in data:
    print(d)
