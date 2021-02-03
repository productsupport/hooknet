def log_command(scriptname, args, logfile):
    # Create dictionary from arguments parser
    args = vars(args)
    # Reconstruct full command
    command = 'python {}'.format(scriptname)
    for k in args.keys():
        command += ' --{} {}'.format(k, args[k])
    # Write command and settings to logfile
    dotted = ''.join(['-' for i in range(60)]) + '\n'
    with open(logfile, 'w') as f:
        f.write('Command\n')
        f.write(dotted + command + '\n\n')
        f.write('\nSettings\n')
        f.write(dotted)
        width = max([len(k) for k in args.keys()])
        for k in args.keys():
            f.write(("{:>" + str(width) + "}: {}\n").format(k, args[k]))

    
