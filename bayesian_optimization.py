from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from scipy.io.idl import AttrDict

from experiment_manager import covid_with_neptune
from models import Actions


def black_box_covid(contrastive_epochs, lr):
    PARAMS = {

        'unite_epochs': 220,
        'contrastive_epochs': int(contrastive_epochs),
        'action': Actions.TrainWithContrastive.value,
        'freeze_base_model': False,
        'contrastive_learning_rate': lr,

    }
    PARAMS['contrastive_model_path'] = f"/home/eyal/privet_dev/cov3_sagi/contrastive_models/contrastive_{PARAMS['contrastive_epochs']}.h5"

    params = AttrDict(PARAMS)
    print(params)
    accuracy = covid_with_neptune(params)
    return accuracy


# Bounded region of parameter space
pbounds = {'contrastive_epochs': (2, 10), 'lr': (0.0000001, 0.0001)}

optimizer = BayesianOptimization(
    f=black_box_covid,
    pbounds=pbounds,
    random_state=1,
)

logger = JSONLogger(path="./logs_opt1.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


optimizer.maximize(
    init_points=10,
    n_iter=200,
)

print(f'best params so far: {optimizer.max}')

