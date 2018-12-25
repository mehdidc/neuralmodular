import numpy as np

def generate_toy(w=28, h=28, ph=(1, 5), pw=(1, 5),
                 nb_patches=1, random_state=None, fg_color=None,
                 bg_color=None, colored=False):
    nb_cols = 3 if colored else 1
    if not bg_color:
        bg_color = [0] * nb_cols
    if not fg_color:
        fg_color = [255] * nb_cols
    rng = np.random.RandomState(random_state)
    ph_ = rng.randint(*ph) if hasattr(ph, '__len__') else ph
    pw_ = rng.randint(*pw) if hasattr(pw, '__len__') else pw
    img = np.ones((h + ph_, w + pw_, nb_cols)) * bg_color
    for _ in range(nb_patches):
        x, y = rng.randint(ph_ / 2, w), rng.randint(pw_ / 2, h)
        img[y:y + pw_, x:x + ph_] = fg_color
    img = img[0:h, 0:w, :]
    return img.astype('uint8')
