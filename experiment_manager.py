import neptune
from scipy.io.idl import AttrDict

import united_training
from models import Actions


def covid_with_neptune(params_):
    neptune.init('eyalasulin/covid',
                 api_token='token_hidden_before_submitting')
    # ## configuration

    neptune.create_experiment(name='contrastive_covid', params=params_)

    loss, accuracy, history, auc = united_training.main(params_)

    if history is not None:
        neptune.log_metric('loss', loss)
        neptune.log_metric('accuracy', accuracy)
        for key in history.history.keys():
            for item in history.history[key]:
                neptune.log_metric(f'h_{key}', item)

    neptune.stop()
    return accuracy


PARAMS = {

    'unite_epochs': 220,
    'contrastive_epochs': 2,
    'action': Actions.TrainWithContrastive.value,
    'freeze_base_model': False,
    'contrastive_learning_rate': 1e-5,

}
PARAMS['contrastive_model_path'] = f"/home/eyal/privet_dev/cov3_sagi/contrastive_models/contrastive_{PARAMS['contrastive_epochs']}.h5"

if __name__ == '__main__':
    params = AttrDict(PARAMS)
    covid_with_neptune(params)

