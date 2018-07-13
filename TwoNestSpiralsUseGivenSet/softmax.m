function a = softmax(z)
   temp = exp(z(1,:)) + exp(z(2,:));
   a(1,:) = exp(z(1,:)) ./ temp; 
   a(2,:) = exp(z(2,:)) ./ temp;
end