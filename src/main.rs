fn main() -> anyhow::Result<()> {
    #[cfg(debug_assertions)]
    {
        extern crate audio_engine;
        use audio_engine::testing::run_cli_simulation;
        use std::fs::File;
        use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
        let file = File::create("output.log").expect("failed to create log file");
        let file_layer = fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .with_thread_names(true)
            .with_level(true);

        let console_layer = fmt::layer()
            .with_writer(std::io::stderr) // stays attached to your console
            .with_ansi(true)
            .with_target(false)
            .with_thread_names(true)
            .with_level(true);

        if !cfg!(feature = "dev-tools") {
            tracing_subscriber::registry()
                .with(console_layer)
                .with(file_layer)
                .init();
        }
        // use audio_engine::AudioEngine;
        // let engine = AudioEngine::new().expect("engine init failed");
        // engine.start_output().expect("output stream failed");
        // let _ = engine
        //     .create_metronome(
        //         120.0,
        //         vec![3, 1, 1],
        //         vec![vec![2], vec![1], vec![1]],
        //         0.8,
        //         true,
        //     )
        //     .expect("met create failed");
        // use audio_engine::analysis::theory::{Interval, Note};
        // let f1 = 440.0;
        // let f2 = 522.0;
        // let n1 = Note::from_freq(f1, None);
        // let n2 = Note::from_freq(f2, None);
        // let int = Interval::new(
        //     &[f1, f2],
        //     Some(audio_engine::analysis::tuner::TuningSystem::Pythagorean),
        // );
        // println!("Note 1: {n1}, Note 2: {n2}, Interval: {int:?}");
        run_cli_simulation();
    }
    Ok(())
}
