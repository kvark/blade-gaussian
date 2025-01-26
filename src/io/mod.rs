mod ply;
mod spz;

pub fn load(file_name: &str) -> crate::Model {
    if file_name.ends_with(".ply") {
        ply::load(file_name)
    } else if file_name.ends_with(".spz") {
        spz::load(file_name)
    } else {
        panic!("Unsupported file name: {}", file_name);
    }
}
