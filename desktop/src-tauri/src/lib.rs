use std::{sync::Mutex, time::Duration};
use tauri::Manager;
use tauri_plugin_shell::{process::CommandChild, ShellExt};

struct SidecarState(Mutex<Option<CommandChild>>);

fn app_ready() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:8501".parse().unwrap(),
        Duration::from_millis(120),
    ).is_ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .setup(|app| {
            let (_rx, child) = app
                .shell()
                .sidecar("englishlearn-server")?
                .args(["--port", "8501"])
                .spawn()?;
            *app.state::<SidecarState>().0.lock().unwrap() = Some(child);

            let handle = app.handle().clone();
            std::thread::spawn(move || {
                for _ in 0..160 {
                    if app_ready() {
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.navigate("http://127.0.0.1:8501".parse().unwrap());
                            let _ = window.set_focus();
                        }
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(125));
                }
                if let Some(window) = handle.get_webview_window("main") {
                    let _ = window.eval("document.querySelector('#error').style.display='block';document.querySelector('#error').textContent='启动超时，请重新打开应用或导出诊断信息。';");
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed) {
                if let Some(child) = window.state::<SidecarState>().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running EnglishLearn desktop");
}
