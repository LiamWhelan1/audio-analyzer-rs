fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32) / (n as f32);
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
        })
        .collect()
}
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}
fn build_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let fmin = fmin;
    let fmax = fmax;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();

    let fft_freqs: Vec<f32> = (0..=n_fft / 2)
        .map(|i| (i as f32) * (sample_rate as f32) / (n_fft as f32))
        .collect();

    let mut filters = vec![vec![0.0f32; n_fft / 2 + 1]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for (k, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_left && freq <= f_center {
                filters[m][k] = (freq - f_left) / (f_center - f_left);
            } else if freq >= f_center && freq <= f_right {
                filters[m][k] = (f_right - freq) / (f_right - f_center);
            } else {
                filters[m][k] = 0.0;
            }
        }
        // normalize filter area to unity (optional, good for energy preservation)
        let enorm: f32 = filters[m].iter().sum();
        if enorm > 0.0 {
            for v in &mut filters[m] {
                *v /= enorm;
            }
        }
    }
    filters
}
fn spect_to_mel(mag: &[f32], mel_filters: &Vec<Vec<f32>>) -> Vec<f32> {
    mel_filters
        .iter()
        .map(|filt| filt.iter().zip(mag.iter()).map(|(a, b)| a * b).sum())
        .collect()
}
fn power_to_db(power: &[f32], amin: f32, top_db: f32) -> Vec<f32> {
    let ref_value = 1.0;
    power
        .iter()
        .map(|p| {
            let val = 10.0 * ((p.max(amin) / ref_value).log10());
            val
        })
        .collect::<Vec<_>>()
        .iter()
        .map(|v| {
            // Clamp relative to max
            *v
        })
        .collect()
}
/// Simple linear mapping for dB to 0..255 (you can change contrast)
fn db_to_u8(db: &[f32], min_db: f32, max_db: f32) -> Vec<u8> {
    db.iter()
        .map(|v| {
            let scaled = ((*v - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
            (scaled * 255.0) as u8
        })
        .collect()
}
