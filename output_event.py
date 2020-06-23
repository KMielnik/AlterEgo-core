import collections

Type = collections.namedtuple('Type', 'isError name text ')


class OutputEvent:
    class Types:
        OPENING_MODEL = Type(
            isError=False, name="OPENING_MODEL", text='Opening model.')
        PROCESSING_STARTED = Type(
            isError=False, name="PROCESSING_STARTED", text='Processing started.')
        OPENING_VIDEO = Type(
            isError=False, name="OPENING_VIDEO", text='Opening video.')
        PREPROCESSING_FIND_BEST_FRAME = Type(
            isError=False, name="PREPROCESSING_FIND_BEST_FRAME", text='Preprocessing frames for --find_best_frame.')
        OPENING_VIDEO_TEMP = Type(
            isError=False, name="OPENING_VIDEO_TEMP", text='Opening preprocessed video from temp.')
        PREPROCESSING_FIND_BEST_FRAME_TEMP = Type(
            isError=False, name="PREPROCESSING_FIND_BEST_FRAME_TEMP", text='Loading preprocessed frames for --find_best_frame from temp.')
        PROCESSING_VIDEO_STARTED = Type(
            isError=False, name="PROCESSING_VIDEO_STARTED", text='Started generating output video.')
        SAVING_OUTPUT_VIDEO = Type(
            isError=False, name="SAVING_OUTPUT_VIDEO", text='Video generated, saving to file.')
        VIDEO_SAVED = Type(isError=False, name="VIDEO_SAVED",
                           text='Generated video has ben saved.')

        ERROR_OPENING_IMAGE = Type(
            isError=True, name="ERROR_OPENING_IMAGE", text='Problem with opening image.')
        ERROR_OPENING_VIDEO = Type(
            isError=True, name="ERROR_OPENING_VIDEO", text='Problem with opening video.')
        ERROR_OPENING_MODEL = Type(
            isError=True, name="ERROR_OPENING_MODEL", text='Problem with opening model.')

    EventType: Type
    Time: float

    def __init__(self, EventType: Type, Time: float):
        self.EventType = EventType
        self.Time = Time
