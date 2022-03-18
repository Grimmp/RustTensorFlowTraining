use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
fn main() {

    //Sigmatures declared when we saved the model
    let train_input_parameter_input_name = "training_input";
    let train_input_parameter_target_name = "training_target";
    let pred_input_parameter_name = "inputs";

    //Names of output nodes of the graph, retrieved with the saved_model_cli command
    let train_output_parameter_name = "output_0";
    let pred_output_parameter_name = "output_0";

    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!
    let input_tensor: Tensor<f32> = Tensor::new(&[1,2]).with_values(&[1.0, 1.0]).unwrap();
    let target_tensor: Tensor<f32> = Tensor::new(&[1,1]).with_values(&[2.0]).unwrap();

    //Path of the saved model
    let save_dir = "create_model/custom_model";

    //Create a graph
    let mut graph = Graph::new();

    //Load save model as graph
    let bundle = SavedModelBundle::load(
        &SessionOptions::new(), &["serve"], &mut graph, save_dir
    ).expect("Can't load saved model");

    //Initiate a session
    let session = &bundle.session;

    //Retrieve the train functions signature
    let signature_train = bundle.meta_graph_def().get_signature("train").unwrap();

    //
    let input_info_train = signature_train.get_input(train_input_parameter_input_name).unwrap();
    let target_info_train = signature_train.get_input(train_input_parameter_target_name).unwrap();

    //
    let output_info_train = signature_train.get_output(train_output_parameter_name).unwrap();

    //
    let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).unwrap();
    let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).unwrap();

    //
    let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).unwrap();

    //The values will be fed to and retrieved from the model with this
    let mut args = SessionRunArgs::new();

    //Feed the tensors into the graph
    args.add_feed(&input_op_train, 0, &input_tensor);
    args.add_feed(&target_op_train, 0, &target_tensor);

    //Fetch result from graph
    let mut out = args.request_fetch(&output_op_train, 0);

    //Run the session
    session
    .run(&mut args)
    .expect("Error occurred during calculations");

    //Retrieve the result of the operation
    let loss: f32 = args.fetch(out).unwrap()[0];

    println!("Loss: {:?}", loss);


    //Retrieve the pred functions signature
    let signature_train = bundle.meta_graph_def().get_signature("pred").unwrap();

    //
    let input_info_pred = signature_train.get_input(pred_input_parameter_name).unwrap();

    //
    let output_info_pred = signature_train.get_output(pred_output_parameter_name).unwrap();

    //
    let input_op_pred = graph.operation_by_name_required(&input_info_pred.name().name).unwrap();

    //
    let output_op_pred = graph.operation_by_name_required(&output_info_pred.name().name).unwrap();

    args.add_feed(&input_op_pred, 0, &input_tensor);

    out = args.request_fetch(&output_op_pred, 0);

    //Run the session
    session
    .run(&mut args)
    .expect("Error occurred during calculations");

    let prediction: f32 = args.fetch(out).unwrap()[0];

    println!("Prediction: {:?}\nActual: 2.0", prediction);
    
   
}

    

    
    
    
    
    
