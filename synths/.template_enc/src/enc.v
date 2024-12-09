module enc #() (
        state,
        index,
        W,
        out
);
    `include "params.v"

    localparam OUTPUT_ADDR_WIDTH = $clog2(OUTPUTS);
    localparam OUTPUT_SIZE = 1 << OUTPUT_ADDR_WIDTH;

    input wire [PBITS-1:0] state;
    input wire [PBITS-1:0] index;
    output wire signed [31:0] out;

    input signed [OUTPUTS*32 - 1:0] W;
    wire signed [OUTPUT_SIZE*32 - 1:0] adder_in;
    wire [OUTPUTS -1:0] z_out, o_out;
    wire [PBITS-1:0] z_state, o_state;

    enc_circuit z_circuit (
        .state(z_state),
        .outputs(z_out)
    );
    enc_circuit o_circuit (
        .state(o_state),
        .outputs(o_out)
    );

    assign o_state = state | index;
    assign z_state = state & ~index;

    weight_mux #(.OUTPUTS(OUTPUTS), .OUTPUT_SIZE(OUTPUT_SIZE)) weight_mux_inst (
        .z_out(z_out),
        .o_out(o_out),
        .W(W),
        .adder_in(adder_in)
    );

    adder_tree #( .N(OUTPUT_SIZE)) adder_inst (
        .inputs(adder_in),
        .out(out)
    );
endmodule
