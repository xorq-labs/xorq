use gbdt::gradient_boost::GBDT;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct SessionModelRegistry {
    pub(crate) models: HashMap<String, Arc<GBDT>>,
}

impl Debug for SessionModelRegistry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionModelRegistry")
            .field("models", &self.models.len())
            .finish()
    }
}

impl SessionModelRegistry {
    pub(crate) fn new() -> Self {
        Default::default()
    }
}

pub trait ModelRegistry {
    fn register_model(&self, name: &str, path: &str, objective: &str);
    fn register_json_model(&self, name: &str, path: &str);
}
