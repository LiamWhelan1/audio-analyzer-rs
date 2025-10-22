use audio_analyzer_rs::testing::run_cli_simulation;
use std::fs::File;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
fn main() -> anyhow::Result<()> {
    let file = File::create("output.log").expect("failed to create log file");
    let file_layer = fmt::layer()
        .with_writer(file)
        .with_ansi(false)
        .with_thread_names(true)
        .with_level(true);

    let console_layer = fmt::layer()
        .with_writer(std::io::stdout) // stays attached to your console
        .with_ansi(true)
        .with_target(false)
        .with_thread_names(true)
        .with_level(true);

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();
    run_cli_simulation();
    Ok(())
}
