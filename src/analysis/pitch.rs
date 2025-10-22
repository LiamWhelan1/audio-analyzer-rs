/// A compact implementation of the YIN pitch algorithm.
/// Reference: A. de Cheveigne and H. Kawahara, 2002.
pub fn yin_pitch(frame: &[f32], samplerate: f32, min_freq: f32, max_freq: f32) -> Option<f32> {
    let n = frame.len();
    let max_tau = (samplerate / min_freq) as usize;
    let min_tau = (samplerate / max_freq) as usize;
    let max_tau = max_tau.min(n / 2);

    let mut diff = vec![0.0f32; max_tau + 1];
    for tau in min_tau..=max_tau {
        let mut s = 0.0f32;
        for i in 0..(n - tau) {
            let d = frame[i] - frame[i + tau];
            s += d * d;
        }
        diff[tau] = s;
    }

    // cumulative mean normalized difference
    let mut cmnd = vec![0.0f32; diff.len()];
    let mut running = 0.0f32;
    for tau in 1..=max_tau {
        running += diff[tau];
        cmnd[tau] = diff[tau] * (tau as f32 / running.max(1e-8));
    }

    // absolute threshold
    let threshold = 0.1f32;
    let mut tau_est = None;
    for tau in min_tau..=max_tau {
        if cmnd[tau] < threshold {
            // parabolic interpolation around tau to refine pitch
            let t = tau as f32;
            let better_tau = {
                let a = cmnd[tau - 1];
                let b = cmnd[tau];
                let c = cmnd.get(tau + 1).copied().unwrap_or(b);
                let denom = a - 2.0 * b + c;
                if denom.abs() < 1e-8 {
                    t
                } else {
                    t + (a - c) / (2.0 * denom)
                }
            };
            tau_est = Some(better_tau);
            break;
        }
    }

    tau_est.map(|tau| samplerate / tau)
}
