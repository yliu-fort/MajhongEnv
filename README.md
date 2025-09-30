# MajhongEnv

An environment and utilities for Mahjong-related experiments.

## GUI Wrappers

The project ships with two optional graphical front-ends for interactive play
and debugging:

* **pygame wrapper** – the legacy implementation located in
  [`src/mahjong_wrapper.py`](src/mahjong_wrapper.py).
* **Kivy wrapper** – a feature-parity port located in
  [`src/mahjong_wrapper_kivy.py`](src/mahjong_wrapper_kivy.py).

### Installing GUI Dependencies

The base environment requirements are listed in [`requirements.txt`](requirements.txt).
To use the GUI front-ends install the optional packages as well:

```bash
pip install -r requirements.txt
```

On some platforms Kivy has additional system-level dependencies. Refer to the
[official installation guide](https://kivy.org/doc/stable/gettingstarted/installation.html)
for platform-specific instructions.

### Running the Kivy Wrapper

Once the optional dependencies are installed you can launch an interactive game
session by swapping in the Kivy wrapper when constructing the environment, for
example:

```python
from mahjong_wrapper_kivy import MahjongEnv

env = MahjongEnv(num_players=4)
env.reset()
while not env.done:
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
```

The module mirrors the pygame wrapper controls: autoplay, pause, single-step,
and status overlays are available via the on-screen buttons. Close the window or
use the `q` key to exit the session.

### Testing

Basic smoke tests for the Kivy wrapper are included. Run the full suite with:

```bash
pytest
```

The GUI-specific tests automatically skip when Kivy is not available.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
