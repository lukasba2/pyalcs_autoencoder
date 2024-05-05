import os
import dill


def get_from_cache_or_run(cache_path, fun):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return dill.load(f)
    else:
        res = fun()

        # save the results to file
        with open(cache_path, 'wb') as f:
            dill.dump(res, f)

        return res
