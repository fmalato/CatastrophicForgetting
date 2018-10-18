""" Some useful code to resize images, as required by the article. For now, I've put it there just to keep in mind
    it actually exists."""

from tensorforce.core.preprocessing import PreprocessorStack, Grayscale, ImageResize, Sequence

pp_gray = Grayscale()  # initialize grayscale preprocessor
pp_seq = Sequence(4)  # initialize sequence preprocessor

stack = PreprocessorStack()  # initialize preprocessing stack
stack.from_spec()  # add grayscale preprocessor to stack
stack.add(pp_seq)  # add maximum preprocessor to stack

state = env.reset()  # reset environment
processed_state = stack.process(state)  # process state

preprocessing_config = [
    {
        "type": "image_resize",
        "width": 84,
        "height": 84
    }, {
        "type": "grayscale"
    }, {
        "type": "center"
    }, {
        "type": "sequence",
        "length": 4
    }
]