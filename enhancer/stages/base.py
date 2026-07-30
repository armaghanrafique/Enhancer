"""Common base class for enhancement stages.

A stage takes an RGB uint8 frame and returns an RGB uint8 frame. Each stage
loads its deep model lazily on first use; if that fails it transparently falls
back to a classical (OpenCV) implementation so the pipeline never hard-crashes.
"""


class Stage:
    name = "base"

    def __init__(self, device, deep=True, **opts):
        self.device = device
        self.deep = deep
        self.opts = opts
        self._model = None
        self._mode = None  # "deep" | "classical", resolved on first process()

    # -- to be implemented by subclasses -----------------------------------
    def _load_model(self):
        """Return a callable model, or None if the deep path is unavailable."""
        return None

    def _process_deep(self, img):
        raise NotImplementedError

    def _process_classical(self, img):
        raise NotImplementedError

    # -- shared dispatch ----------------------------------------------------
    def _ensure_mode(self):
        if self._mode is not None:
            return
        if self.deep:
            try:
                self._model = self._load_model()
            except Exception as e:  # noqa: BLE001
                print(f"  ! [{self.name}] deep model load failed ({e}); using classical")
                self._model = None
            self._mode = "deep" if self._model is not None else "classical"
        else:
            self._mode = "classical"
        print(f"  - [{self.name}] mode: {self._mode}")

    def process(self, img):
        self._ensure_mode()
        if self._mode == "deep":
            return self._process_deep(img)
        return self._process_classical(img)
