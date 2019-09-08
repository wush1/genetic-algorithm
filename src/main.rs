extern crate nalgebra as na;

use na::{DMatrix, Complex};

struct NeuralNetwork {
    input_layer_size: usize,
    hidden_layer_size: usize,
    output_layer_size: usize,

    weights1: DMatrix<f32>,
    weights2: DMatrix<f32>,
}

impl NeuralNetwork {
    fn forward(&self, y: Vec<f32>) -> Vec<f32> {
        if y.len() != self.input_layer_size {
            panic!("Number of inputs don't match input size");
        }

        let x = DMatrix::from_row_slice(
            self.input_layer_size, 1, &y,);

        let z2 = &self.weights1 * x;
        let a2 = z2.map(|x| self.sigmoid(x));

        let z3 = &self.weights2 * a2;
        let y_hat = z3.map(|x| self.sigmoid(x));

        z3.as_slice().to_vec()
    }
    fn mutate(&mut self, amount: f32) {
        self.weights1 = self.weights1.map(|x| x + amount);
        self.weights2 = self.weights2.map(|x| x + amount);
    }
    fn sigmoid(&self, i: f32) -> f32 {
        1.0 / (1.0 + i.exp())
    }
}

fn new_network
(input_layer_size: usize, hidden_layer_size: usize, output_layer_size: usize) -> NeuralNetwork {
    NeuralNetwork {
        input_layer_size: input_layer_size,
        hidden_layer_size: hidden_layer_size,
        output_layer_size: output_layer_size,

        weights1: DMatrix::new_random(hidden_layer_size, input_layer_size),
        weights2: DMatrix::new_random(output_layer_size, hidden_layer_size),
        
    }
}

fn main() {
    let mut network = new_network(3, 4, 2);
    println!("{:?}", network.forward(vec![0.5, 1.0, 1.0]));
    println!("{:?}", network.forward(vec![0.5, 1.0, 1.0]));
    println!("{:?}", network.forward(vec![0.6, 0.53, 0.0]));
    network.mutate(3.0);
    println!("{:?}", network.forward(vec![0.5, 1.0, 1.0]));
}
