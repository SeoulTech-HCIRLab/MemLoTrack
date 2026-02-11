import torch.nn as nn
from typing import Any, Optional
from trackit.data.protocol.eval_input import TrackerEvalData


class TrackerEvaluator:
    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def begin(self, data: TrackerEvalData):
        raise NotImplementedError()

    def prepare_initialization(self) -> Any:
        raise NotImplementedError()

    def on_initialized(self, model_init_output: Any):
        raise NotImplementedError()

    def prepare_tracking(self) -> Any:
        raise NotImplementedError()

    def on_tracked(self, model_track_outputs: Any):
        raise NotImplementedError()

    def do_custom_update(self, compiled_model: Any, raw_model: Optional[nn.Module]):
        raise NotImplementedError()

    def end(self) -> Any:
        raise NotImplementedError()



"""
run_tracker_evaluator 마지막 부분에서 'x_feat'를 함께 반환하도록 수정한 코드
"""
def run_tracker_evaluator(tracker_evaluator, data, optimized_model, raw_model):
    if data is None:
        print("[Debug] Data is None")
        return None

    # 1) begin
    tracker_evaluator.begin(data)

    # 2) prepare_initialization => init_outputs
    init_params = tracker_evaluator.prepare_initialization()
    init_outputs = optimized_model(init_params) if init_params is not None else None
    tracker_evaluator.on_initialized(init_outputs)

    # 3) prepare_tracking => track_outputs
    track_params = tracker_evaluator.prepare_tracking()
    track_outputs = optimized_model(track_params) if track_params is not None else None

    # => pipeline.on_tracked(track_outputs) 내부에서 x_feat가 들어갈 수도 있음
    tracker_evaluator.on_tracked(track_outputs)

    # 4) do_custom_update
    tracker_evaluator.do_custom_update(optimized_model, raw_model)

    # 5) end => end_result
    end_result = tracker_evaluator.end()
    # end_result typically: {'evaluated_sequences':..., 'evaluated_frames':...}
    # Debug statements to check values
    # print("[Debug] end_result:", end_result)
    # print("[Debug] track_outputs:", track_outputs)
    
    # Ensure end_result is a dictionary
    if end_result is None:
        end_result = {}

    # --------------------------
    # (B) 병합 로직:
    # 'track_outputs'에 'x_feat'가 있다면, end_result와 합쳐서 반환
    merged_result = {}
    merged_result.update(end_result)  # => evaluated_sequences, evaluated_frames

    if track_outputs is not None and isinstance(track_outputs, dict):
        if 'x_feat' in track_outputs:
            merged_result['x_feat'] = track_outputs['x_feat']
    # print("[Debug] merged_result:", merged_result)

    return merged_result