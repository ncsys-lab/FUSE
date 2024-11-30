module enc #() (
        state,
        index,
        W,
        out
);
    `include "params.v"

    //localparam OUTPUTS = (N*(N - 1)) >> 1;
    //localparam PBITS = N * $clog2(N) - N + 1;
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

    genvar i;
    generate
        for(i = 0; i < OUTPUTS; i = i + 1) begin
            assign adder_in[i*32+:32] = (z_out[i] ^ o_out[i]) ? ((z_out[i]) ? W[i*32+:32]: -W[i*32+:32]):0;
        end
        for(i = OUTPUTS; i < OUTPUT_SIZE; i = i + 1) begin
            assign adder_in[i*32+:32] = 0;
        end
    endgenerate

    adder_tree #( .N(OUTPUT_SIZE)) adder_inst (
        .inputs(adder_in),
        .out(out)
    );
    // Output of the adder tree is the final inter-stage sum
endmodule
