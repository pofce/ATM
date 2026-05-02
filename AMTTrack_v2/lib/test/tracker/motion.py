import numpy as np


class KalmanMotionModel:
    """
    Constant-velocity Kalman filter for bounding box tracking.

    State vector  x: [cx, cy, vx, vy, w, h]
    Observation   z: [cx, cy, w, h]   (derived from input [x1, y1, w, h])

    Transition model (F):
        cx  ← cx + vx * dt
        cy  ← cy + vy * dt
        vx  ← vx
        vy  ← vy
        w   ← w
        h   ← h

    dt is the real inter-frame interval in seconds, derived from the
    sequence frame rate (e.g. 1.0 s for 1 Hz, 0.5 s for 2 Hz).
    Velocity process noise is scaled by dt² so that position uncertainty
    grows correctly as the filter runs in predict-only mode.
    """

    # Base noise parameters (at dt = 1 s).
    # Q_vel is the velocity process noise variance per second²;
    # it is multiplied by dt² so units stay consistent.
    _Q_pos = 1.0    # position process noise (cx, cy)
    _Q_vel = 10.0   # velocity process noise coefficient (vx, vy)
    _Q_sz  = 1.0    # size process noise (w, h)

    _R_diag = np.array([1.0, 1.0,    # cx, cy  measurement noise
                        10.0, 10.0], # w,  h   measurement noise
                       dtype=np.float64)

    def __init__(self):
        # Observation matrix: z = H @ x
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 4] = 1.0  # w
        self.H[3, 5] = 1.0  # h

        self.R = np.diag(self._R_diag)

        self.x = None   # state mean      (6,)
        self.P = None   # state covariance (6,6)
        self._last_dt = 1.0  # remembered for predict_only() when no dt given

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _xywh_to_cxcywh(box):
        """[x1, y1, w, h] → [cx, cy, w, h]"""
        x1, y1, w, h = box
        return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float64)

    @staticmethod
    def _cxcywh_to_xywh(cx, cy, w, h):
        """[cx, cy, w, h] → [x1, y1, w, h]"""
        return [cx - 0.5 * w, cy - 0.5 * h, w, h]

    def _make_F(self, dt):
        """Build the state transition matrix for a given dt (seconds)."""
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt   # cx += vx * dt
        F[1, 3] = dt   # cy += vy * dt
        return F

    def _make_Q(self, dt):
        """Build the process noise covariance for a given dt."""
        return np.diag([
            self._Q_pos,           # cx
            self._Q_pos,           # cy
            self._Q_vel * dt**2,   # vx  (noise scales with dt²)
            self._Q_vel * dt**2,   # vy
            self._Q_sz,            # w
            self._Q_sz,            # h
        ])

    def _is_initialised(self):
        return self.x is not None

    # ------------------------------------------------------------------ #
    # public API                                                           #
    # ------------------------------------------------------------------ #
    def update(self, state, dt=1.0):
        """
        Ingest a new [x1, y1, w, h] observation and run a full
        predict → correct Kalman cycle.

        On the first call the filter is initialised from the observation
        (velocity set to zero, high initial uncertainty).

        Args:
            state: list or array [x1, y1, w, h] in image coordinates.
            dt:    inter-frame interval in seconds (default 1.0).
                   For a 2 Hz sequence pass 0.5; for 4 Hz pass 0.25.
        """
        z = self._xywh_to_cxcywh(state)
        self._last_dt = dt

        if not self._is_initialised():
            self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]],
                               dtype=np.float64)
            # High initial covariance for velocities; position/size from obs.
            self.P = np.diag([10.0, 10.0, 100.0, 100.0, 10.0, 10.0])
            return

        F = self._make_F(dt)
        Q = self._make_Q(dt)

        # --- predict step ---
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # --- update step ---
        y = z - self.H @ x_pred                      # innovation (4,)
        S = self.H @ P_pred @ self.H.T + self.R       # (4,4)
        K = P_pred @ self.H.T @ np.linalg.inv(S)     # (6,4) Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

    def predict_only(self, dt=None):
        """
        Advance the state by one time step without an observation.
        Intended for use during blind periods (burst / low-confidence)
        where the network output is unreliable.

        Args:
            dt: inter-frame interval in seconds.  Defaults to the dt
                used in the most recent update() call.

        Returns:
            list [x1, y1, w, h] — predicted bounding box.
        """
        if not self._is_initialised():
            raise RuntimeError(
                "KalmanMotionModel.predict_only() called before any update().")

        if dt is None:
            dt = self._last_dt

        F = self._make_F(dt)
        Q = self._make_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        cx, cy, _vx, _vy, w, h = self.x
        w = max(w, 1.0)
        h = max(h, 1.0)
        return self._cxcywh_to_xywh(cx, cy, w, h)

    def reset(self):
        """Clear filter state (call when target is re-acquired after long loss)."""
        self.x = None
        self.P = None
        self._last_dt = 1.0
