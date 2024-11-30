module conv (
        state,
        index,
        W,
        out
);
    `include "params.v"
    input wire [PBITS-1:0] state;
    input wire signed [31:0] index;
    input wire signed [PBITS*32 - 1:0] W;
    output wire signed [31:0] out;

    localparam ADDER_SIZE = 1 << $clog2(PBITS);
    wire signed [ADDER_SIZE*32 - 1:0] adder_in;

    genvar i;
    generate
    for (i = 0; i < PBITS; i = i + 1) begin
        assign adder_in[i*32+:32] = (state[i] | (i == index)) ? W[i*32+:32] : '0;
    end
    for(i = PBITS; i < ADDER_SIZE; i = i + 1) begin
        assign adder_in[i*32+:32] = 0;
    end
    endgenerate

    adder_tree #( .N(ADDER_SIZE)) adder_inst (
        .inputs(adder_in),
        .out(out)
    );

endmodule
