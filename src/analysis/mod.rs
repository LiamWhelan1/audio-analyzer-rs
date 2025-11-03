pub mod chroma;
pub mod onset;
pub mod pitch;
use std::sync::{Arc, Mutex};

pub enum TuningSystem {
    EqualTemperament,
    JustIntonation,
}

#[derive(Clone, Debug)]
pub struct ChannelAnalysis {
    pub mean: f64,
    pub variance: f64,
    pub kurtosis: f64,
    pub correlation: f64,
}

#[derive(Clone, Debug)]
pub enum AudioState {
    Silence,
    WhiteNoise,
    Structured,
}

pub fn analyze_channels(
    samples: &[f32],
    num_channels: usize,
) -> (Vec<ChannelAnalysis>, AudioState) {
    let mut channel_data = vec![Vec::new(); num_channels];

    // Split interleaved samples by channel
    for (i, sample) in samples.iter().enumerate() {
        channel_data[i % num_channels].push(*sample);
    }

    let mut analyses = Vec::new();
    let mut active_channels = 0;

    for ch in &channel_data {
        if ch.is_empty() {
            analyses.push(ChannelAnalysis {
                mean: 0.0,
                variance: 0.0,
                kurtosis: 0.0,
                correlation: 0.0,
            });
            continue;
        }

        let mean = ch.iter().map(|&x| x as f64).sum::<f64>() / ch.len() as f64;
        let variance = ch.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / ch.len() as f64;

        // Excess kurtosis: distinguishes noise vs structured
        let kurtosis = ch.iter().map(|&x| (x as f64 - mean).powi(4)).sum::<f64>()
            / (ch.len() as f64 * variance.powi(2))
            - 3.0;

        // Correlation: structure detection (lag-1 autocorr)
        let mut corr_num = 0.0;
        for i in 1..ch.len() {
            corr_num += (ch[i - 1] as f64 - mean) * (ch[i] as f64 - mean);
        }
        let corr = corr_num / ((ch.len() - 1) as f64 * variance.max(1e-12));

        if variance > 1e-6 {
            active_channels += 1;
        }

        analyses.push(ChannelAnalysis {
            mean,
            variance,
            kurtosis,
            correlation: corr,
        });
    }

    // Decide global state
    let avg_var = analyses.iter().map(|a| a.variance).sum::<f64>() / analyses.len() as f64;
    let avg_corr =
        analyses.iter().map(|a| a.correlation.abs()).sum::<f64>() / analyses.len() as f64;
    let avg_kurt = analyses.iter().map(|a| a.kurtosis).sum::<f64>() / analyses.len() as f64;

    let state = if avg_var < 1e-7 {
        AudioState::Silence
    } else if avg_corr < 0.05 && avg_kurt.abs() < 1.0 {
        AudioState::WhiteNoise
    } else {
        AudioState::Structured
    };

    (analyses, state)
}

// Example integration with your CPAL callback:
pub fn handle_audio_input(data: &[f32], channels: usize) {
    let (analyses, state) = analyze_channels(data, channels);

    println!("\n--- AUDIO ANALYSIS ---");
    for (i, a) in analyses.iter().enumerate() {
        println!(
            "CH{}: mean={:.3e}, var={:.3e}, corr={:.3e}, kurt={:.3e}",
            i, a.mean, a.variance, a.correlation, a.kurtosis
        );
    }

    println!("Detected channels: {}", analyses.len());
    println!(
        "Active channels: {}",
        analyses.iter().filter(|a| a.variance > 1e-6).count()
    );
    println!("Audio state: {:?}", state);
}
