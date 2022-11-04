# SpMV


## Changing power modes in NVIDIA Xavier AGX

To learn with power mode is activated 
```
sudo /usr/sbin/nvpmodel -q
```

To switch the power modes 
```
sudo /usr/sbin/nvpmodel -m x
```
where x is a number in the range [0,7]
