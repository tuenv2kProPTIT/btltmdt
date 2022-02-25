from dataclasses import dataclass
from tfdet.core.config import CoreConfig
from tfdet.core.utils import iou
from tfdet.utils import shape_utils as shape_utils
import tensorflow as tf 
@dataclass
class MaxIOUAssignerConfig(CoreConfig):
    pos_iou_thr: float = 0.5
    neg_iou_thr: float = 0.4
    min_pos_iou: float = 0.
    iou_calculator: str = "IouSimilarity"


class MaxIOUAssigner:
    def __init__(self, cfg:MaxIOUAssignerConfig, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.cfg = cfg 

    def match(self, anchors, targets, ignore_tagets):
        ''' anchors: N,4
            targets: M,4
            ignore_targets: M :
                +> 0: ignore
                +> 1: don't ignore
            -----------------------
            return tensor: shape[N]: where 0<=index[i] < M
        '''
        if self.cfg.iou_calculator=='IouSimilarity':
            match = iou(targets, anchors)
        else:
            raise ValueError(self.cfg.iou_calculator)
        return self._match(match, ignore_tagets)
        
    def _match(self, similarity_matrix, valid_rows):
        """Tries to match each column of the similarity matrix to a row.
        Args:
        similarity_matrix: tensor of shape [N, M] representing any similarity
            metric.
        Returns:
        Match object with corresponding matches for each of M columns.
        """
        def _match_when_rows_are_empty():
            """Performs matching when the rows of similarity matrix are empty.
            When the rows are empty, all detections are false positives. So we return
            a tensor of -1's to indicate that the columns do not match to any rows.
            Returns:
                matches:  int32 tensor indicating the row each column matches to.
            """
            similarity_matrix_shape = shape_utils.shape_list(
                similarity_matrix)
            return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

        def _match_when_rows_are_non_empty():
            """Performs matching when the rows of similarity matrix are non empty.
            Returns:
                matches:  int32 tensor indicating the row each column matches to.
            """
            # Matches for each column
            matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)

            matched_vals = tf.reduce_max(similarity_matrix, 0)
            below_unmatched_threshold = tf.greater(self.cfg.neg_iou_thr,
                                                matched_vals)

            between_thresholds = tf.logical_and(
                tf.greater_equal(matched_vals, self.cfg.neg_iou_thr),
                tf.greater(self.cfg.pos_iou_thr, matched_vals))
            
            matches = self._set_values_using_indicator(matches,
                                                below_unmatched_threshold,
                                                -1)
            matches = self._set_values_using_indicator(matches,
                                                between_thresholds,
                                                -2)
              

            if self._force_match_for_each_row:
                similarity_matrix_shape = shape_utils.shape_list(
                    similarity_matrix)
                force_match_column_ids = tf.argmax(similarity_matrix, 1,
                                           output_type=tf.int32)
          
                # tf.print(temp,one_h.shape, temp * one_h)
                force_match_column_indicators = (
                    tf.one_hot(
                        force_match_column_ids, depth=similarity_matrix_shape[1]) *
                    tf.cast(tf.expand_dims(valid_rows, axis=-1), dtype=tf.float32))
                force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                                output_type=tf.int32)
                # tf.print(force_match_row_ids, tf.where(force_match_row_ids>0))
                force_match_column_mask = tf.cast(
                    tf.reduce_max(force_match_column_indicators, 0), tf.bool)
                final_matches = tf.where(force_match_column_mask,
                                        force_match_row_ids, matches)

                return final_matches
            else:

                return matches

        if similarity_matrix.shape.is_fully_defined():
            if similarity_matrix.shape[0] == 0:
                return _match_when_rows_are_empty()
            else:
                return _match_when_rows_are_non_empty()
        else:
            return tf.cond(
                tf.greater(tf.shape(similarity_matrix)[0], 0),
                _match_when_rows_are_non_empty, _match_when_rows_are_empty)

    def _set_values_using_indicator(self, x, indicator, val):
        """Set the indicated fields of x to val.
        Args:
            x: tensor.
            indicator: boolean with same shape as x.
            val: scalar with value to set.
        Returns:
            modified tensor.
        """
        indicator = tf.cast(indicator, x.dtype)
        return x * (1 - indicator) + val * indicator