import marimo

__generated_with = "0.8.22"
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
    mo.md("""We read the file and get a small chunk to show. We also get the metadata about the chunk""")
    return


@app.cell
def _(wavfile):
    SAMPLE_RATE, data = wavfile.read('./input/BadApple.wav')
    data = data[:, 1]
    chunk_size = len(data) // 10
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    data = chunks[0]
    DURATION = len(data) / SAMPLE_RATE
    SAMPLES = len(data)
    return DURATION, SAMPLES, SAMPLE_RATE, chunk_size, chunks, data


@app.cell
def _(mo):
    mo.md("""Get the graph of the signal as of function of time in seconds""")
    return


@app.cell
def _(DURATION, SAMPLE_RATE, data, np):
    times = np.linspace(0, DURATION, int(DURATION*SAMPLE_RATE))
    signal_values = [data[int(time * SAMPLE_RATE) - 1] for time in times]
    return signal_values, times


@app.cell
def _(mo):
    mo.md("""Compute the fourier transform""")
    return


@app.cell
def _(data, np, plt, signal_values, times):
    plt.plot(times, signal_values)
    plt.title("The audio chunks represented of a signal of the seconds domain")
    plt.show()
    fft = np.fft.fft(data)
    fft = fft[:len(fft) // 2]
    fft = abs(fft)
    return (fft,)


@app.cell
def _(mo):
    mo.md(
        """
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
def _(SAMPLE_RATE, data, fft, find_peaks, np, plt):
    frequencies = np.fft.fftfreq(data.size, d=1 / SAMPLE_RATE)[:data.size // 2]
    print(len(frequencies))
    peaks, props = find_peaks(fft)
    # plt.plot(np.linspace(0, len(fft), len(fft)), fft)
    # plt.grid(True)
    freqs = []
    for peak in peaks:
        plt.text(peak, fft[peak], f'{peak:.2f}', ha='center', fontsize=8)
        freqs.append(frequencies[peak])
    print(f'Freqs: {freqs}')
    times_ran = 0
    from numba import njit
    @njit
    def func(x):
        ans  = 0

        for peak in peaks:
            ans += np.sin(x * 2 * np.pi * frequencies[peak])
        return ans
    # plt.show()
    # plt.plot(times, [func(x) for x in times])
    # print(times_ran)
    # plt.title("Reconstructed wave")
    # plt.show()
    return freqs, frequencies, func, njit, peak, peaks, props, times_ran


@app.cell
def _(mo):
    mo.md("""As you can see, the wave isn't very accurate, let's try a different method. We'll sample $\\omega$ amount of equidistant points from the graph""")
    return


@app.cell
def _(DURATION, SAMPLE_RATE, func, mo, njit, np, plt, times, wavfile):
    amount_slider = mo.ui.slider(1, 1000, label="$\\omega$", show_value=True)
    amount_slider
    _times = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
    @njit(parallel=True)
    def test(times):
        result = []
        for time in times:
            result.append(func(time))
        return result
    good = test(times)
    wavfile.write("test.wav", SAMPLE_RATE,  np.array(good))

    plt.plot(times, good)
    return amount_slider, good, test


@app.cell
def _(SAMPLE_RATE, data, good, wavfile):
    wavfile.write("test2.wav", SAMPLE_RATE, data)
    print(len(good))
    print(len(data))
    return


@app.cell
def _(amount_slider, fft, np):
    freqs2 = []
    amount = amount_slider.value
    distance_between_points = np.floor(len(fft) / amount)
    print(f"Sampling ~{amount} points ~{distance_between_points} apart")
    return amount, distance_between_points, freqs2


@app.cell
def _():
    import marimo as mo
    import marimo
    from scipy.io import wavfile
    from scipy.signal import find_peaks, butter, find_peaks_cwt
    import numpy as np
    import matplotlib.pyplot as plt
    import cupy as cp
    return (
        butter,
        cp,
        find_peaks,
        find_peaks_cwt,
        marimo,
        mo,
        np,
        plt,
        wavfile,
    )


if __name__ == "__main__":
    app.run()
