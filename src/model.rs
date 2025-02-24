pub trait ModelRegistry {
    fn register_model(&self, name: &str, path: &str, objective: &str);
    fn register_json_model(&self, name: &str, path: &str);
}
