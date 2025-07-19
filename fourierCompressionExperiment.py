import marimo

__generated_with = "0.14.11"
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
    data = data[:, 1] # the first channel
    return SAMPLE_RATE, data


@app.cell
def _(data, mo):
    amount_of_chunks = mo.ui.slider(1, len(data)//100, label="Amount of chunks", value=len(data)//100)
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
    mo.md("""Get the graph of the signal as of function of time in seconds""")
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
    mo.md("""Compute the fourier transform""")
    return


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
def _(SAMPLE_RATE, chunk, find_peaks, np, plt):

    fft = np.fft.fft(chunk)
    fft = fft[:len(fft) // 2]
    frequencies = np.fft.fftfreq(chunk.size, d=1 / SAMPLE_RATE)[:chunk.size // 2]
    plt.title("The fourier transform of the signal")
    plt.plot(frequencies, fft)
    plt.show()
    peaks, props = find_peaks(fft)
    return fft, frequencies, peaks


@app.cell
def _(fft, frequencies, njit, np, peaks, plt):

    freqs = []
    for peak in peaks:
        plt.text(peak, fft[peak], f'{peak:.2f}', ha='center', fontsize=8)
        freqs.append(frequencies[peak])
    plt.title("The fourier transform of the signal (with freqs)")
    plt.plot(frequencies, fft)
    plt.show()
    times_ran = 0
    @njit
    def reconstructed_wave(x):
        ans  = 0

        for peak in peaks:
            ans += np.sin(x * 2 * np.pi * frequencies[peak])
        return ans
    # plt.show()
    # plt.plot(times, [reconstructed_wave(x) for x in times])
    # print(times_ran)
    # plt.title("Reconstructed wave")
    # plt.show()
    return (reconstructed_wave,)


@app.cell
def _(mo):
    mo.md("""As you can see, the wave isn't very accurate, let's try a different method. We'll sample $\\omega$ amount of equidistant points from the graph""")
    return


@app.cell
def _(
    SAMPLE_RATE,
    njit,
    np,
    plt,
    reconstructed_wave,
    signal_values,
    times,
    wavfile,
):

    # TODO: MAKE A KERNEL FOR THIS
    @njit()
    def reconstruct(times):
        result = []
        for time in times:
            result.append(reconstructed_wave(time))
        return result
    new_wave = reconstruct(times)
    wavfile.write("reconstructed_wave.wav", SAMPLE_RATE,  np.array(new_wave))
    plt.figure(figsize=(20,6))
    plt.title("The reconstructed wave")
    plt.plot(times, new_wave, color="blue", label="Reconstucted wave")
    plt.plot(times, signal_values, color="red", label="Original wave")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md("""As you can see, we have a small problem. We didn't account for the __*strength*__ of the frequencies, we assumed their all equal which is almost never the case. Let's fix that.""")
    return


@app.cell
def _(fft, frequencies, np, peaks, plt):

    better_result_freqs : dict[float, float] = {} 
    for _freq in frequencies:
        index = np.where(frequencies == _freq)

        better_result_freqs[fft[index[0]][0]] = _freq
    print(peaks)
    print(better_result_freqs)
    plt.title("Peaks and strengths ")
    plt.xlabel("Freqs")
    plt.ylabel("Strengths")
    plt.plot(better_result_freqs.values(), better_result_freqs.keys())
    plt.show()
    max_value = max(better_result_freqs.keys())
    better_result_freqs = {k / max_value : v for k, v in better_result_freqs.items()}
    print(better_result_freqs)
    def better_reconstructed_wave(x):
        ans  = 0
        for strength, freq in better_result_freqs.items():
        
            ans += np.real(strength) * np.sin(x * 2 * np.pi * freq + np.imag(strength))
        return ans
    return (better_reconstructed_wave,)


@app.cell
def _(better_reconstructed_wave, np, plt, signal_values, times):

    # TODO: MAKE A KERNEL FOR THIS

    def better_reconstruct(times):
        _result = []
        for _time in times:
            _result.append(better_reconstructed_wave(_time))
        return _result
    _new_wave = better_reconstruct(times)
    # wavfile.write("reconstructed_wave.wav", SAMPLE_RATE,  np.array(_new_wave))
    # wavfile.write("old bad wave.wav", SAMPLE_RATE,  np.array(signal_values))
    normalised_signal_values = normalise(np.array(signal_values))
    _new_wave = normalise(np.array(_new_wave))
    plt.figure(figsize=(20,6))
    plt.title("The reconstructed wave")
    plt.plot(times, _new_wave, color="blue", label="Reconstucted wave")
    plt.plot(times, normalised_signal_values, color="red", label="Original wave")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md("Now we do that for all the chunks")
    return


@app.cell
def _():
    final_wave = []
    return (final_wave,)


@app.cell
def _():

    
    return


app._unparsable_cell(
    r"""
    d = len(data) / SAMPLE_RATE
    s = SAMPLE_RATE * d

    ts = np.linspace(0, d, int(d*SAMPLE_RATE))

    print(type(final_wave[0]))

    final_wave2 = []
    for arr in final_wave:
        if type(arr) == list:
            for val in arr:  # ✅ just iterate over arr directly
                final_wave2.append(val)
        else:
            final_wave2.append(arr)
    total_duration = len(final_wave2) / SAMPLE_RATE
    good_i = 0
    for i in final_wave2:
    
        if i != 0:
            good_i = final_wave2.index(i)
            break;
    print(len(final_wave2[good_i:]))
    ts = np.linspace(0, total_duration, len(final_wave2))
    plt.plot(times, , label=\"FInal\")
    plt.show()
    wavfile.write(\"test.wav\", SAMPLE_RATE, np.array(final_wave2))
    """,
    name="_"
)


@app.cell
def _(chunks, mo):
    sy = mo.ui.slider(1, len(chunks), label="Which chunk to choose2")
    sy
    return (sy,)


@app.cell
def _(SAMPLE_RATE, final_wave, np, plt, sy):

    good_chunk = final_wave[sy.value]

    tsy = np.linspace(0, len(good_chunk) / SAMPLE_RATE, len(good_chunk))
    plt.plot(tsy, good_chunk)
    plt.title(sy.value)
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
    return find_peaks, mo, njit, np, plt, wavfile


if __name__ == "__main__":
    app.run()
