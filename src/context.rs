use crate::catalog::{PyCatalog, PyTable};
use crate::dataframe::PyDataFrame;
use crate::dataset::Dataset;
use crate::errors::DataFusionError;
use crate::expr::sort_expr::PySortExpr;
use crate::functions::greatest::GreatestFunc;
use crate::functions::hash_int::HashIntFunc;
use crate::functions::least::LeastFunc;
use crate::ibis_table::IbisTable;
use crate::object_storage::{
    get_object_store, register_object_store_and_config_extensions, AwsOptions, GcpOptions,
};
use crate::optimizer::PyOptimizerRule;
use crate::provider::PyTableProvider;
use crate::py_record_batch_provider::PyRecordBatchProvider;
use crate::udaf::PyAggregateUDF;
use crate::udf::PyScalarUDF;
use crate::udwf::PyWindowUDF;
use crate::utils::wait_for_future;
use arrow::array::RecordBatchReader;
use arrow_ord::sort::SortOptions;
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::ffi_stream::ArrowArrayStreamReader;
use datafusion::arrow::pyarrow::PyArrowType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::MemTable;
use datafusion::datasource::TableProvider;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_expr::{expressions, LexOrdering, PhysicalSortExpr};
use datafusion::prelude::{CsvReadOptions, DataFrame, ParquetReadOptions};
use datafusion_common::config::ConfigFileType;
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use datafusion_expr::ScalarUDF;
use object_store::{Error, ObjectMeta, ObjectStore, ObjectStoreScheme};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

/// Configuration options for a SessionContext
#[pyclass(name = "SessionConfig", module = "let", subclass)]
#[derive(Clone, Default)]
pub(crate) struct PySessionConfig {
    pub(crate) config: SessionConfig,
}

impl From<SessionConfig> for PySessionConfig {
    fn from(config: SessionConfig) -> Self {
        Self { config }
    }
}

#[pymethods]
impl PySessionConfig {
    #[pyo3(signature = (config_options=None))]
    #[new]
    fn new(config_options: Option<HashMap<String, String>>) -> Self {
        let mut config = SessionConfig::new();
        if let Some(hash_map) = config_options {
            for (k, v) in &hash_map {
                config = config.set(k, &ScalarValue::Utf8(Some(v.clone())));
            }
        }

        Self { config }
    }

    fn with_create_default_catalog_and_schema(&self, enabled: bool) -> Self {
        Self::from(
            self.config
                .clone()
                .with_create_default_catalog_and_schema(enabled),
        )
    }

    fn with_default_catalog_and_schema(&self, catalog: &str, schema: &str) -> Self {
        Self::from(
            self.config
                .clone()
                .with_default_catalog_and_schema(catalog, schema),
        )
    }

    fn with_information_schema(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_information_schema(enabled))
    }

    fn with_batch_size(&self, batch_size: usize) -> Self {
        Self::from(self.config.clone().with_batch_size(batch_size))
    }

    fn with_target_partitions(&self, target_partitions: usize) -> Self {
        Self::from(
            self.config
                .clone()
                .with_target_partitions(target_partitions),
        )
    }

    fn with_repartition_aggregations(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_repartition_aggregations(enabled))
    }

    fn with_repartition_joins(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_repartition_joins(enabled))
    }

    fn with_repartition_windows(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_repartition_windows(enabled))
    }

    fn with_repartition_sorts(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_repartition_sorts(enabled))
    }

    fn with_repartition_file_scans(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_repartition_file_scans(enabled))
    }

    fn with_repartition_file_min_size(&self, size: usize) -> Self {
        Self::from(self.config.clone().with_repartition_file_min_size(size))
    }

    fn with_parquet_pruning(&self, enabled: bool) -> Self {
        Self::from(self.config.clone().with_parquet_pruning(enabled))
    }

    fn set(&self, key: &str, value: &str) -> Self {
        Self::from(self.config.clone().set_str(key, value))
    }
}

#[pyclass(name = "SessionState", module = "let", subclass)]
#[derive(Clone)]
pub(crate) struct PySessionState {
    pub(crate) session_state: SessionState,
}

impl From<SessionState> for PySessionState {
    fn from(session_state: SessionState) -> Self {
        Self { session_state }
    }
}

#[pymethods]
impl PySessionState {
    #[pyo3(signature = (config=None))]
    #[new]
    fn new(config: Option<PySessionConfig>) -> Self {
        let config = if let Some(c) = config {
            c.config
        } else {
            SessionConfig::default().with_information_schema(true)
        };
        let runtime_config = RuntimeEnvBuilder::default();
        let runtime = Arc::new(runtime_config.build().unwrap());

        let session_state = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(runtime)
            .with_default_features()
            .build();

        Self { session_state }
    }

    fn add_optimizer_rule(&mut self, rule: &Bound<'_, PyAny>) -> Self {
        let rule = PyOptimizerRule::new(rule);

        Self::from(
            SessionStateBuilder::new_from_existing(self.session_state.clone())
                .with_optimizer_rule(Arc::new(rule))
                .build(),
        )
    }
}

/// `PySessionContext` is able to plan and execute DataFusion plans.
/// It has a powerful optimizer, a physical planner for local execution, and a
/// multithreaded execution engine to perform the execution.
#[pyclass(name = "SessionContext", module = "let", subclass)]
#[derive(Clone)]
pub(crate) struct PySessionContext {
    pub(crate) ctx: SessionContext,
}

#[pymethods]
impl PySessionContext {
    #[pyo3(signature = (session_state=None, config=None))]
    #[new]
    fn new(
        session_state: Option<PySessionState>,
        config: Option<PySessionConfig>,
    ) -> PyResult<Self> {
        let runtime_config = RuntimeEnvBuilder::default();
        let runtime = Arc::new(runtime_config.build()?);
        let session_state = match (session_state, config) {
            (Some(s), _) => s.session_state,
            (None, Some(c)) => SessionStateBuilder::new()
                .with_config(c.config)
                .with_runtime_env(runtime)
                .with_default_features()
                .build(),
            (None, _) => {
                let session_config = SessionConfig::default().with_information_schema(true);
                SessionStateBuilder::new()
                    .with_config(session_config)
                    .with_runtime_env(runtime)
                    .with_default_features()
                    .build()
            }
        };

        let ctx = SessionContext::new_with_state(session_state.clone());
        // register the UDF with the context, so it can be invoked by name and from SQL
        ctx.register_udf(ScalarUDF::from(GreatestFunc::new()));
        ctx.register_udf(ScalarUDF::from(LeastFunc::new()));
        ctx.register_udf(ScalarUDF::from(HashIntFunc::new()));

        Ok(PySessionContext { ctx })
    }

    /// Returns a PyDataFrame whose plan corresponds to the SQL statement.
    fn sql(&mut self, query: &str, py: Python) -> PyResult<PyDataFrame> {
        let result = self.ctx.sql(query);
        let df = wait_for_future(py, result)?;
        Ok(PyDataFrame::new(df))
    }

    fn deregister_table(&mut self, name: &str) -> PyResult<()> {
        self.ctx
            .deregister_table(name)
            .map_err(DataFusionError::from)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name,
                        path,
                        table_partition_cols=vec![],
                        parquet_pruning=true,
                        file_extension=".parquet",
                        skip_metadata=true,
                        schema=None,
                        storage_options=None
    ))]
    fn register_parquet(
        &mut self,
        name: &str,
        path: &str,
        table_partition_cols: Vec<(String, String)>,
        parquet_pruning: bool,
        file_extension: &str,
        skip_metadata: bool,
        schema: Option<PyArrowType<Schema>>,
        storage_options: Option<HashMap<String, String>>,
        py: Python,
    ) -> PyResult<()> {
        let mut options = ParquetReadOptions::default()
            .table_partition_cols(convert_table_partition_cols(table_partition_cols)?)
            .parquet_pruning(parquet_pruning)
            .skip_metadata(skip_metadata);
        options.file_extension = file_extension;
        options.schema = schema.as_ref().map(|x| &x.0);
        let storage_options = storage_options.unwrap_or_default();

        let result = async {
            register_object_store_and_config_extensions(
                &self.ctx,
                &path.to_string(),
                Some(ConfigFileType::PARQUET),
                &storage_options,
            )
            .await?;
            self.ctx.register_parquet(name, path, options).await
        };

        wait_for_future(py, result).map_err(DataFusionError::from)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name,
                        path,
                        schema=None,
                        has_header=true,
                        delimiter=",",
                        schema_infer_max_records=1000,
                        file_extension=".csv",
                        file_compression_type=None,
                        storage_options=None))]
    fn register_csv(
        &mut self,
        name: &str,
        path: PathBuf,
        schema: Option<PyArrowType<Schema>>,
        has_header: bool,
        delimiter: &str,
        schema_infer_max_records: usize,
        file_extension: &str,
        file_compression_type: Option<String>,
        storage_options: Option<HashMap<String, String>>,
        py: Python,
    ) -> PyResult<()> {
        let path = path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("Unable to convert path to a string"))?;
        let delimiter = delimiter.as_bytes();
        if delimiter.len() != 1 {
            return Err(PyValueError::new_err(
                "Delimiter must be a single character",
            ));
        }
        let storage_options = storage_options.unwrap_or_default();

        let mut options = CsvReadOptions::new()
            .has_header(has_header)
            .delimiter(delimiter[0])
            .schema_infer_max_records(schema_infer_max_records)
            .file_extension(file_extension)
            .file_compression_type(parse_file_compression_type(file_compression_type)?);
        options.schema = schema.as_ref().map(|x| &x.0);

        let result = async {
            register_object_store_and_config_extensions(
                &self.ctx,
                &path.to_string(),
                Some(ConfigFileType::CSV),
                &storage_options,
            )
            .await?;
            self.ctx.register_csv(name, path, options).await
        };

        wait_for_future(py, result).map_err(DataFusionError::from)?;

        Ok(())
    }

    fn register_record_batches(
        &mut self,
        name: &str,
        partitions: PyArrowType<Vec<Vec<RecordBatch>>>,
    ) -> PyResult<()> {
        let schema = partitions.0[0][0].schema();
        let table = MemTable::try_new(schema, partitions.0)?;
        self.ctx
            .register_table(name, Arc::new(table))
            .map_err(DataFusionError::from)?;
        Ok(())
    }

    #[pyo3(signature = (name,
                        reader,
                        sort_order=None))]
    pub fn register_record_batch_reader(
        &mut self,
        name: &str,
        reader: PyArrowType<ArrowArrayStreamReader>,
        sort_order: Option<Vec<PySortExpr>>,
    ) -> PyResult<()> {
        let reader = reader.0;
        let schema = reader.schema();

        let mut ordering = LexOrdering::default();
        if let Some(exprs) = sort_order {
            for sort in exprs {
                match sort.sort.expr {
                    Expr::Column(col) => match expressions::col(&col.name, schema.deref()) {
                        Ok(expr) => {
                            ordering.push(PhysicalSortExpr {
                                expr,
                                options: SortOptions {
                                    descending: !sort.sort.asc,
                                    nulls_first: sort.sort.nulls_first,
                                },
                            });
                        }
                        Err(_) => break,
                    },
                    expr => {
                        return Err(PyValueError::new_err(format!("Unsupported expr {expr:?}")));
                    }
                }
            }
        }
        let table = PyRecordBatchProvider::new(reader, ordering.clone());
        self.ctx
            .register_table(name, Arc::new(table))
            .map_err(DataFusionError::from)?;

        Ok(())
    }

    pub fn register_ibis_table(
        &mut self,
        name: &str,
        reader: &Bound<'_, PyAny>,
        py: Python,
    ) -> PyResult<()> {
        let table: Arc<dyn TableProvider> = Arc::new(IbisTable::new(reader, py)?);

        self.ctx
            .register_table(name, table)
            .map_err(DataFusionError::from)?;

        Ok(())
    }

    #[pyo3(name = "register_table_provider")]
    pub fn register_py_table_provider(
        &mut self,
        name: &str,
        provider: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let provider = PyTableProvider::new(provider)?;
        let table: Arc<dyn TableProvider> = Arc::new(provider);

        self.ctx
            .register_table(name, table)
            .map_err(DataFusionError::from)?;

        Ok(())
    }

    pub fn register_dataframe(&mut self, name: &str, dataframe: PyDataFrame) -> PyResult<()> {
        let table: Arc<dyn TableProvider> = dataframe.df.as_ref().clone().into_view();

        self.ctx
            .register_table(name, table)
            .map_err(DataFusionError::from)?;

        Ok(())
    }

    pub fn register_table(&mut self, name: &str, table: &PyTable) -> PyResult<()> {
        self.ctx
            .register_table(name, table.table())
            .map_err(DataFusionError::from)?;
        Ok(())
    }

    fn register_dataset(&self, name: &str, dataset: &Bound<'_, PyAny>, py: Python) -> PyResult<()> {
        let table: Arc<dyn TableProvider> = Arc::new(Dataset::new(dataset, py)?);

        self.ctx
            .register_table(name, table)
            .map_err(DataFusionError::from)?;

        Ok(())
    }

    fn tables(&self) -> HashSet<String> {
        self.ctx
            .catalog("datafusion")
            .unwrap()
            .schema("public")
            .unwrap()
            .table_names()
            .iter()
            .cloned()
            .collect()
    }

    fn table(&self, name: &str, py: Python) -> PyResult<PyDataFrame> {
        let x = wait_for_future(py, self.ctx.table(name)).map_err(DataFusionError::from)?;
        Ok(PyDataFrame::new(x))
    }

    fn table_exist(&self, name: &str) -> PyResult<bool> {
        Ok(self.ctx.table_exist(name)?)
    }

    fn empty_table(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.ctx.read_empty()?))
    }

    fn session_id(&self) -> String {
        self.ctx.session_id()
    }

    fn register_udf(&mut self, udf: PyScalarUDF) -> PyResult<()> {
        self.ctx.register_udf(udf.function);
        Ok(())
    }

    fn register_udaf(&mut self, udaf: PyAggregateUDF) -> PyResult<()> {
        self.ctx.register_udaf(udaf.function);
        Ok(())
    }

    pub fn register_udwf(&mut self, udwf: PyWindowUDF) -> PyResult<()> {
        self.ctx.register_udwf(udwf.function);
        Ok(())
    }

    #[pyo3(signature = (path, file_format, **kwargs))]
    pub fn get_object_metadata(
        &mut self,
        path: PathBuf,
        file_format: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
        py: Python,
    ) -> PyResult<Py<PyDict>> {
        let path = path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("Unable to convert path to a string"))?;

        let storage_options = kwargs
            .unwrap_or(&PyDict::new(py))
            .get_item("storage_options")
            .and_then(|item| item.unwrap_or(PyDict::new(py).into_any()).extract())
            .unwrap_or_default();

        let location = &path.to_string();

        // Parse the location URL to extract the scheme and other components
        let table_path = ListingTableUrl::parse(location)?;

        // Extract the scheme (e.g., "s3", "gcs") from the parsed URL
        let scheme = table_path.scheme();

        // Obtain a reference to the URL
        let url = table_path.as_ref();

        match scheme {
            // For Amazon S3 or Alibaba Cloud OSS
            "s3" | "oss" | "cos" => {
                // Register AWS specific table options in the session context:
                self.ctx
                    .register_table_options_extension(AwsOptions::default())
            }
            // For Google Cloud Storage
            "gs" | "gcs" => {
                // Register GCP specific table options in the session context:
                self.ctx
                    .register_table_options_extension(GcpOptions::default())
            }
            // For unsupported schemes, do nothing:
            _ => {}
        }

        let format = match file_format {
            "csv" => Some(ConfigFileType::CSV),
            "json" => Some(ConfigFileType::JSON),
            "parquet" => Some(ConfigFileType::PARQUET),
            _ => None,
        };

        // Clone and modify the default table options based on the provided options
        let mut table_options = self.ctx.state().default_table_options().clone();
        if let Some(format) = format {
            table_options.set_config_format(format);
        }

        if let "s3" | "oss" | "cos" | "gs" | "gcs" = scheme {
            table_options.alter_with_string_hash_map(&storage_options)?;
        }

        let result = async {
            // Retrieve the appropriate object store based on the scheme, URL, and modified table options
            let store = get_object_store(&self.ctx.state(), scheme, url, &table_options)
                .await
                .map_err(|e| Error::Generic {
                    store: "Config",
                    source: format!("Source: {e}").into(),
                })?;
            let (_, object_path) = ObjectStoreScheme::parse(url)?;
            store.head(&object_path).await
        };

        let metadata =
            wait_for_future(py, result).map_err(|e| PyValueError::new_err(format!("Err: {e}")))?;

        metadata_to_pydict(py, metadata)
    }

    fn __repr__(&self) -> PyResult<String> {
        let config = self.ctx.copied_config();
        let mut config_entries = config
            .options()
            .entries()
            .iter()
            .filter(|e| e.value.is_some())
            .map(|e| format!("{} = {}", e.key, e.value.as_ref().unwrap()))
            .collect::<Vec<_>>();
        config_entries.sort();
        Ok(format!(
            "SessionContext: id={}; configs=[\n\t{}]",
            self.session_id(),
            config_entries.join("\n\t")
        ))
    }

    #[pyo3(signature = (name="datafusion"))]
    fn catalog(&self, name: &str) -> PyResult<PyCatalog> {
        match self.ctx.catalog(name) {
            Some(catalog) => Ok(PyCatalog::new(catalog)),
            None => Err(PyKeyError::new_err(format!(
                "Catalog with name {} doesn't exist.",
                &name,
            ))),
        }
    }
}

impl PySessionContext {
    async fn _table(&self, name: &str) -> datafusion_common::Result<DataFrame> {
        self.ctx.table(name).await
    }
}

impl From<PySessionContext> for SessionContext {
    fn from(ctx: PySessionContext) -> SessionContext {
        ctx.ctx
    }
}

fn convert_table_partition_cols(
    table_partition_cols: Vec<(String, String)>,
) -> Result<Vec<(String, DataType)>, DataFusionError> {
    table_partition_cols
        .into_iter()
        .map(|(name, ty)| match ty.as_str() {
            "string" => Ok((name, DataType::Utf8)),
            _ => Err(DataFusionError::Common(format!(
                "Unsupported data type '{ty}' for partition column"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()
}

fn parse_file_compression_type(
    file_compression_type: Option<String>,
) -> Result<FileCompressionType, PyErr> {
    FileCompressionType::from_str(&*file_compression_type.unwrap_or("".to_string()).as_str())
        .map_err(|_| {
            PyValueError::new_err("file_compression_type must one of: gzip, bz2, xz, zstd")
        })
}

/// Convert ObjectMeta to a Python dictionary
fn metadata_to_pydict(py: Python, metadata: ObjectMeta) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    dict.set_item("location", metadata.location.to_string())?;
    dict.set_item("last_modified", metadata.last_modified.to_string())?;
    dict.set_item("size", metadata.size)?;
    dict.set_item("e_tag", metadata.e_tag.unwrap_or_default())?;

    // Add any version information if available
    if let Some(version) = metadata.version {
        dict.set_item("version", version)?;
    }

    Ok(dict.into())
}
