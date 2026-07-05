"""Data stream container for LiteDBX data managers."""

import logging
import random

from .ldb_data import LdbData

logger = logging.getLogger(__name__)


class DataStream(list[LdbData]):
    """
    Ordered stream of raw LiteDBX input data batches.
    
    Basic schema: [ LdbData, ...<streams> ]
    """

    def init(
        self, complete_dataset: LdbData, dynamic_steps: list[float], seed: int
    ) -> None:
        """Initialize stream partitions from a complete dataset."""
        if len(self) > 0:
            raise RuntimeError("Data stream has already been initialized.")

        logger.info("Start data stream construction.")
        partitions = self._build(complete_dataset, dynamic_steps, seed)
        self.extend(partitions)
        logger.info("Data stream construction completed.")

    @staticmethod
    def _build(
        complete_dataset: LdbData, dynamic_steps: list[float], seed: int
    ) -> list[LdbData]:
        """Build data stream partitions without mutating this container."""
        total_rows = len(complete_dataset.df)
        indices = list(range(total_rows))
        rng = random.Random(seed)
        rng.shuffle(indices)

        steps = [0] + dynamic_steps
        data_ladder = [int(total_rows * step) for step in steps]

        data_stream = []
        for i in range(1, len(data_ladder)):
            selected_indices = indices[data_ladder[i - 1] : data_ladder[i]]
            df = (
                complete_dataset.df.iloc[selected_indices]
                .copy()
                .reset_index(drop=True)
            )
            data_stream.append(LdbData(df=df, config=complete_dataset.config))

        logger.debug(
            "Built %s data stream partitions from %s rows.",
            len(data_stream),
            total_rows,
        )
        return data_stream
