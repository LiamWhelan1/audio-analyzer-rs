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
        run_cli_simulation();
    }
    Ok(())
}
