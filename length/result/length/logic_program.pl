length(X,Y) :- cons(W,X)& length(Z,Y)& 
length(X,Y) :- length(X,Z)& length(W,Y)& 
length(X,Y) :- cons(X,Z)& length(X,W)& 
length(X,Y) :- succ(Y,W)& cons(X,W)& 
length(X,Y) :- succ(Y,Z)& succ(Z,Y)& succ(W,Y)& 
length(X,Y) :- succ(Y,W)& succ(W,Y)& cons(X,W)& length(W,Y)& 
length(X,Y) :- succ(W,Y)& cons(Z,W)& length(Z,Y)& length(W,Y)& 
length(X,Y) :- succ(Y,W)& succ(Z,W)& succ(W,Y)& cons(W,X)& cons(W,Z)& length(W,Y)& 
length(X,Y) :- succ(Y,W)& succ(W,Y)& cons(W,X)& length(W,Y)& 
length(X,Y) :- succ(W,Y)& cons(W,X)& length(W,Y)& length(W,Z)& 
[(0.1, 9, 90), (0.1, 10, 100), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1), (-0.0, 0, -1)]
