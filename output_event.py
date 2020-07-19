import collections
import copy

Type = collections.namedtuple('Type', 'IsError Name Text ')


class OutputEvent:
    class Types:
        OPENING_MODEL = Type(
            IsError=False, Name="OPENING_MODEL", Text='Opening model.')
        PROCESSING_STARTED = Type(
            IsError=False, Name="PROCESSING_STARTED", Text='Processing started.')
        OPENING_VIDEO = Type(
            IsError=False, Name="OPENING_VIDEO", Text='Opening video.')
        VIDEO_OPENED = Type(
            IsError=False, Name="VIDEO_OPENED", Text='Driving video opened.')
        PREPROCESSING_FIND_BEST_FRAME = Type(
            IsError=False, Name="PREPROCESSING_FIND_BEST_FRAME", Text='Preprocessing frames for --find_best_frame.')
        OPENING_VIDEO_TEMP = Type(
            IsError=False, Name="OPENING_VIDEO_TEMP", Text='Opening preprocessed video from temp.')
        PREPROCESSING_FIND_BEST_FRAME_TEMP = Type(
            IsError=False, Name="PREPROCESSING_FIND_BEST_FRAME_TEMP", Text='Loading preprocessed frames for --find_best_frame from temp.')
        PROCESSING_VIDEO_STARTED = Type(
            IsError=False, Name="PROCESSING_VIDEO_STARTED", Text='Started generating output video.')
        SAVING_OUTPUT_VIDEO = Type(
            IsError=False, Name="SAVING_OUTPUT_VIDEO", Text='Video generated, saving to file.')
        VIDEO_SAVED = Type(IsError=False, Name="VIDEO_SAVED",
                           Text='Generated video has ben saved.')

        ERROR_OPENING_IMAGE = Type(
            IsError=True, Name="ERROR_OPENING_IMAGE", Text='Problem with opening image.')
        ERROR_OPENING_VIDEO = Type(
            IsError=True, Name="ERROR_OPENING_VIDEO", Text='Problem with opening video.')
        ERROR_OPENING_MODEL = Type(
            IsError=True, Name="ERROR_OPENING_MODEL", Text='Problem with opening model.')
        ERROR_ARGUMENT_PARSING = Type(
            IsError=True, Name="ERROR_ARGUMENT_PARSING", Text='Problem with parsing arguments.')

    EventType: Type
    Filename: str
    Time: float

    def __init__(self, EventType: Type, Time: float, Filename: str = None):
        self.EventType = EventType
        self.Time = Time
        self.Filename = Filename

    def _asdict(self):
        eventDict = copy.deepcopy(self.__dict__)
        eventDict["EventType"] = eventDict["EventType"]._asdict()

        return eventDict
