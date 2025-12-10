import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # Main idea
    An audio file format that stores audio using fourier transforms to decompose the audio into its various signals and then storing the properties of the waves.

    ## Steps to convert (from a wav file):
    - Splits the audio into chunks (to make calculations faster)\
    - Calculates the the fourier transform of each chunks
    - Get each freqency from each wave from the chunk
    - Write it to a file

    ## To reconstruct
    - Sum up the waves given their frequencies
    - Do an inverse fourier transform
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""We read the file and get a small chunk to show. We also get the metadata about the chunk""")
    return


@app.cell
def _(wavfile):
    SAMPLE_RATE, data = wavfile.read('./input/BadApple.wav')
    data = data[:, 1] # the first channel
    return SAMPLE_RATE, data


@app.cell
def _(data, mo):
    amount_of_chunks = mo.ui.slider(2, len(data)//100, label="Amount of chunks", value=5)
    amount_of_chunks
    return (amount_of_chunks,)


@app.cell
def _(amount_of_chunks, data):
    chunk_size = len(data) // amount_of_chunks.value
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return (chunks,)


@app.cell
def _(chunks, mo):
    which_chunk_slider = mo.ui.slider(1, len(chunks), label="Which chunk to choose")
    which_chunk_slider
    return (which_chunk_slider,)


@app.cell
def _(SAMPLE_RATE, chunks, which_chunk_slider):
    chunk = chunks[which_chunk_slider.value]
    DURATION = len(chunk) / SAMPLE_RATE
    SAMPLES = len(chunk)
    return DURATION, chunk


@app.cell
def _(mo):
    mo.md(r"""Get the graph of the signal as of function of time in seconds""")
    return


@app.cell
def _(DURATION, SAMPLE_RATE, chunk, np, plt):
    times = np.linspace(0, DURATION, int(DURATION*SAMPLE_RATE))
    signal_values = [chunk[int(time * SAMPLE_RATE) - 1] for time in times]
    plt.plot(times, signal_values)
    plt.title("The audio chunk represented of a signal of the seconds domain")
    plt.show()
    return signal_values, times


@app.cell
def _(mo, signal_values):
    mo.md(f"""There are {len(signal_values)} samples""")
    return


@app.cell
def _(mo):
    mo.md(r"""Compute the fourier transform""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Compute the actual freqeuncies. \n
    1. We use `np.fftfreq` the get the frequencies that could appear in the fourier transform (i think) 
    2. We find the peaks. The peaks correspond to the most prevalent frequencies 
    3. We graph the fft along with the peaks
    4. We append the frequency at `peak` to the `result` array
    5. We reconstruct the signal using the `result`
    """
    )
    return


@app.cell
def _(SAMPLE_RATE, chunk, np, plt):
    fft = np.fft.rfft(chunk)

    frequencies = np.fft.fftfreq(chunk.size, d=1 / SAMPLE_RATE)[:(chunk.size // 2)+1]
    plt.title("The fourier transform of the signal")
    plt.plot(frequencies, fft)
    plt.show()

    return (fft,)


@app.cell
def _(chunk, fft, np, plt, times):
    plt.title("FFT vs. Normal")
    plt.plot(times, chunk, label="Orginal")
    plt.plot(times, np.fft.irfft(fft), label="Fourier transformed, then inverse Fourier Transformed")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""As you can see, they're the same! The fourier transform is reversible""")
    return


@app.cell
def _(mo):
    mo.md("""We attempt to smooth it""")
    return


@app.cell
def _(chunk, fft, find_peaks, njit, np, plt, times):

    @njit
    def smooth(full_fft, peaks):


        smooth_fft  = [value if list(full_fft).index(value) in peaks else 0 for value in full_fft]
        return smooth_fft

    def smooth_fft(fft):
        peaks = find_peaks(fft)[0]
        return smooth(fft, peaks)

    smoothed = smooth_fft(fft)
    audio_signal = np.fft.irfft(smoothed)
    # plt.plot(times, normalise(smoothed), label="smoothed")
    plt.plot(times, normalise(audio_signal), label="compressed")
    plt.plot(times, normalise(chunk), label="origami")
    plt.legend()


    return (audio_signal,)


@app.cell
def _(mo):
    mo.md("""# The final result!!!""")
    return


@app.cell
def _(SAMPLE_RATE, audio_signal, mo):
    mo.audio(audio_signal, rate=SAMPLE_RATE)
    return


@app.function
# HELPERS:
def normalise(arr):
    max_value = max(arr)
    return arr/max_value


@app.cell
def _():
    import marimo as mo
    import marimo
    from scipy.io import wavfile
    from scipy.signal import find_peaks, butter, find_peaks_cwt
    import numpy as np
    import matplotlib.pyplot as plt
    from numba import njit
    import random as rdm
    return find_peaks, mo, njit, np, plt, wavfile


if __name__ == "__main__":
    app.run()
